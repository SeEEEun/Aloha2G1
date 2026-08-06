#!/usr/bin/env python3
"""Phase-dependent phone grasp/lift/rotate feasibility gate for episode 49.

This program deliberately stops before temporal IK when the physical phone in
the fixed MagSafe scene is outside the conservative Dex3 fingertip reach
envelope.  It never starts Isaac Lab or robot hardware.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

import retarget_episode49_optimized_action_to_g1 as latest  # noqa: E402
import retarget_episode49_relative_bimanual_neutral_pinch_to_g1 as neutral  # noqa: E402

SOURCE = ROOT / (
    "converted_runs/smolvla_20k_episode49_consensus_relative_g1/"
    "g1_episode49_consensus_relative_trajectory.npz"
)
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
ROBOT_POSE = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
MAGNET = ROOT / "isaaclab_magsafe_fixed_scene/magnet_config_v2.json"
OUT_ROOT = ROOT / (
    "converted_runs/smolvla_20k_episode49_phone_grasp_lift_rotate_g1"
)
REPORT = OUT_ROOT / "g1_episode49_phone_grasp_lift_rotate_report.json"
TRAJECTORY = OUT_ROOT / "g1_episode49_phone_grasp_lift_rotate_trajectory.npz"
MARGINS = (0.05, 0.03)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--layout", type=Path, default=LAYOUT)
    p.add_argument("--robot-pose", type=Path, default=ROBOT_POSE)
    p.add_argument("--magnet-config", type=Path, default=MAGNET)
    p.add_argument("--report", type=Path, default=REPORT)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--reach-maxiter", type=int, default=200)
    return p.parse_args()


def minimum_jerk(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10*u**3 - 15*u**4 + 6*u**5


def hysteretic_closed(width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    closure = np.clip((0.044-width)/0.044, 0.0, 1.0)
    state = bool(closure[0] >= 0.65)
    closed = np.empty(len(width), bool)
    for i, value in enumerate(closure):
        if state and value <= 0.45:
            state = False
        elif not state and value >= 0.65:
            state = True
        closed[i] = state
    return closure, closed


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.diff(padded)
    return [(int(a), int(b-1)) for a, b in
            zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def automatic_phases(raw: np.ndarray, midpoint: np.ndarray) -> dict:
    """Infer the manipulation run by bilateral closure and vertical lift."""
    lc, lclosed = hysteretic_closed(raw[:, 6])
    rc, rclosed = hysteretic_closed(raw[:, 13])
    candidates = []
    for start, end in runs(lclosed & rclosed):
        if end-start < 10:
            continue
        peak = start + int(np.argmax(midpoint[start:end+1, 2]))
        lift = float(midpoint[peak, 2]-midpoint[start, 2])
        candidates.append({"start": start, "end": end, "peak": peak,
                           "lift_signal_m": lift, "length": end-start+1})
    if not candidates:
        raise RuntimeError("No bilateral hysteretic closure interval")
    selected = max(candidates, key=lambda x: (x["lift_signal_m"], x["length"]))
    c, e, peak = selected["start"], selected["end"], selected["peak"]
    # PREGRASP starts at the last bilateral closure <=0.65 crossing before
    # contact. No frame number is embedded in this rule.
    pre_candidates = np.flatnonzero(
        (np.minimum(lc, rc)[:-1] < 0.65) &
        (np.minimum(lc, rc)[1:] >= 0.65)
    )
    pre = int(pre_candidates[pre_candidates < c][-1]+1) if np.any(pre_candidates < c) else c
    approach = max(0, pre-max(1, peak-c))
    clearance_signal = midpoint[c:peak+1, 2]-midpoint[c, 2]
    threshold = min(0.03, max(0.01, 0.5*float(clearance_signal.max(initial=0))))
    hit = np.flatnonzero(clearance_signal >= threshold)
    rotate = c+int(hit[0]) if len(hit) else peak
    labels = np.full(len(raw), "APPROACH", dtype="<U16")
    labels[pre:c] = "PREGRASP"
    labels[c:min(c+2, len(raw))] = "CONTACT_PINCH"
    labels[min(c+2, len(raw)):rotate] = "LIFT"
    labels[rotate:e+1] = "CARRY_ROTATE"
    if e+1 < len(raw):
        labels[e+1:] = "PLACE_RELEASE"
    return {
        "labels": labels, "left_closure": lc, "right_closure": rc,
        "candidates": candidates, "selected": selected,
        "boundaries": {"approach": approach, "pregrasp": pre, "contact": c,
                       "lift_clearance": rotate, "carry_end": e,
                       "place_release": min(e+1, len(raw)-1)},
    }


def scene_frames(layout: dict, robot_cfg: dict, magnet: dict,
                 info: dict) -> dict:
    phone = layout["phone"]; table = layout["table"]; charger = layout["charger"]
    lean = math.radians(float(magnet["initial_assembly"]
                                  ["lean_degrees_about_world_x"]))
    phone_rot_world = Rotation.from_rotvec([lean, 0, 0]).as_matrix()
    pivot = np.array([
        0.5*(phone["bottom_left_xy"][0]+phone["bottom_right_xy"][0]),
        phone["bottom_left_xy"][1], table["surface_height"],
    ], float)
    unrotated = pivot + np.array([0, 0, 0.5*phone["size_landscape_xyz"][2]])
    phone_center_world = pivot + phone_rot_world @ (unrotated-pivot)
    axes_world = {
        "long": phone_rot_world[:, 0],
        "thickness": phone_rot_world[:, 1],
        "short": phone_rot_world[:, 2],
        "screen_normal": -(phone_rot_world[:, 1]),
    }
    pose = robot_cfg["g1"]
    robot_rot_world = Rotation.from_quat(
        np.asarray(pose["orientation_wxyz"])[[1, 2, 3, 0]]
    ).as_matrix()
    stand_base_z = float(info["stand_qpos"][2])
    robot_translation = np.array([
        pose["position_xyz_m"][0], pose["position_xyz_m"][1],
        pose["position_xyz_m"][2]-stand_base_z,
    ])
    world_to_g1 = robot_rot_world.T
    phone_center_g1 = world_to_g1 @ (phone_center_world-robot_translation)
    phone_rot_g1 = world_to_g1 @ phone_rot_world

    mount_h = (charger["mount_plate"]["size_xyz"][2]
               if charger["mount_plate"]["enabled"] else 0.0)
    origin = np.array([*charger["center_xy"],
                       table["surface_height"]+mount_h], float)
    tilt = math.radians(float(charger["pad_tilt_degrees_up"]))
    pad_normal = np.array([0, -math.cos(tilt), math.sin(tilt)])
    pad_radius = 0.5*charger["pad_diameter"]
    pad_center = origin + np.array([
        0, charger["pad_center_y_offset"],
        charger["total_height"]-pad_radius*math.cos(tilt),
    ])
    face_center = pad_center + pad_normal*(
        0.5*charger["pad_thickness"]+0.0003+
        magnet["charger_target"]["surface_clearance_m"]
    )
    charger_rot_world = Rotation.from_quat(
        np.asarray(magnet["charger_target"]["target_rotation_wxyz"])
        [[1, 2, 3, 0]]
    ).as_matrix()
    return {
        "phone_center_world": phone_center_world,
        "phone_rotation_world": phone_rot_world, "phone_axes_world": axes_world,
        "table_normal_world": np.array([0., 0., 1.]),
        "robot_rotation_world": robot_rot_world,
        "robot_translation_world_from_mujoco": robot_translation,
        "phone_center_g1": phone_center_g1,
        "phone_rotation_g1": phone_rot_g1,
        "charger_face_center_world": face_center,
        "charger_normal_world": pad_normal,
        "charger_rotation_world": charger_rot_world,
        "charger_center_g1": world_to_g1 @ (face_center-robot_translation),
        "charger_rotation_g1": world_to_g1 @ charger_rot_world,
    }


def phone_nearest_forward(frames: dict, dimensions: np.ndarray) -> float:
    half = 0.5*dimensions
    corners = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1])
                        for z in (-half[2], half[2])])
    world = frames["phone_center_g1"] + corners @ frames["phone_rotation_g1"].T
    return float(world[:, 0].min())


def fingertip_reach_bound(info: dict, margin: float, maxiter: int) -> dict:
    """Maximise a conservative sphere bound of thumb/index collision meshes."""
    layout, schema = neutral.hand_joint_schema(info)
    del schema
    model = layout["model"]; data = mujoco.MjData(model)
    result = {}
    for side, arm_offset in (("left", 0), ("right", 7)):
        gids = []
        for gid in range(model.ngeom):
            body = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[gid])) or ""
            if body.startswith(f"{side}_hand_") and (
                    "thumb_2" in body or "index_1" in body):
                gids.append(gid)
        arm_lo = info["joint_limits"][arm_offset:arm_offset+7, 0]+margin
        arm_hi = info["joint_limits"][arm_offset:arm_offset+7, 1]-margin
        hand_lo = layout["hands"][side]["ranges"][:, 0]
        hand_hi = layout["hands"][side]["ranges"][:, 1]
        if np.any(arm_lo >= arm_hi):
            raise RuntimeError(f"Joint margin {margin} empties arm bounds")

        def objective(x: np.ndarray) -> float:
            data.qpos[:] = model.key_qpos[0]
            data.qpos[info["arm_qpos_ids"][arm_offset:arm_offset+7]] = x[:7]
            data.qpos[layout["hands"][side]["qadr"]] = x[7:]
            mujoco.mj_forward(model, data)
            # rbound encloses the entire mesh.  This over-estimates, rather than
            # under-estimates, physical reach and is therefore a safe blocker.
            return -max(float(data.geom_xpos[g, 0]+model.geom_rbound[g])
                        for g in gids)

        solved = differential_evolution(
            objective, list(zip(np.r_[arm_lo, hand_lo], np.r_[arm_hi, hand_hi])),
            seed=3, popsize=12, maxiter=maxiter, polish=True, workers=1,
            tol=1e-7,
        )
        result[side] = {
            "maximum_conservative_fingertip_forward_reach_m": -float(solved.fun),
            "arm_q": solved.x[:7].tolist(), "dex3_q": solved.x[7:].tolist(),
            "optimizer_success": bool(solved.success),
            "optimizer_message": str(solved.message),
            "fingertip_geom_ids": gids,
        }
    return {"model": layout["model"], "sides": result}


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def main() -> int:
    a = arguments()
    for path in (a.source, a.layout, a.robot_pose, a.magnet_config,
                 latest.G1_XML):
        if not path.exists():
            raise FileNotFoundError(path)
    with np.load(a.source, allow_pickle=False) as z:
        raw = z["optimized_action"].astype(float)
        midpoint = z["aloha_midpoint"].astype(float)
    if raw.shape != (990, 14) or not np.isfinite(raw).all():
        raise RuntimeError(f"optimized_action must be finite [990,14], got {raw.shape}")
    layout = json.loads(a.layout.read_text())
    robot_cfg = json.loads(a.robot_pose.read_text())
    magnet = json.loads(a.magnet_config.read_text())
    info = latest.ik.validate_model(latest.G1_XML)
    phase = automatic_phases(raw, midpoint)
    frames = scene_frames(layout, robot_cfg, magnet, info)
    phone_dims = np.asarray(layout["phone"]["size_landscape_xyz"], float)
    nearest = phone_nearest_forward(frames, phone_dims)
    audits = []
    selected_margin = None
    for margin in MARGINS:
        reach = fingertip_reach_bound(info, margin, a.reach_maxiter)
        gap = {side: nearest-rec["maximum_conservative_fingertip_forward_reach_m"]
               for side, rec in reach["sides"].items()}
        passed = all(value <= 0 for value in gap.values())
        audits.append({"joint_margin_rad": margin, "passed": passed,
                       "required_phone_nearest_forward_m": nearest,
                       "unreachable_gap_m": gap, "reach": reach["sides"]})
        if passed:
            selected_margin = margin
            break

    contact_reachable = selected_margin is not None
    if contact_reachable:
        # The present implementation is a fail-closed endpoint gate.  Reaching
        # here means the temporal contact/collision solver must be run; do not
        # silently emit an unverified trajectory.
        verdict = "TEMPORAL_PHONE_SOLVER_NOT_IMPLEMENTED"
        reason = ("Conservative contact reach gate passed, but a verified "
                  "contact/collision temporal solution has not been produced.")
    else:
        verdict = "PHONE_CONTACT_REACH_CONFLICT"
        reason = (
            "The physical phone's nearest point is beyond the independently "
            "maximized thumb/index mesh reach for both hands even with the "
            "minimum allowed 0.03 rad arm/wrist joint margin. Because this "
            "conservative mesh sphere bound over-estimates actual reach, "
            "CONTACT/PINCH is infeasible before any phase-dependent rotation."
        )
    report = {
        "verdict": verdict, "safety_pass": False, "trajectory_generated": False,
        "videos_generated": False, "stop_phase": "CONTACT_PINCH",
        "stop_reason": reason,
        "source": str(a.source.resolve()), "g1_xml": str(latest.G1_XML),
        "scene_layout": str(a.layout.resolve()),
        "robot_pose_source": str(a.robot_pose.resolve()),
        "magnet_config_source": str(a.magnet_config.resolve()),
        "coordinate_conversion": {
            "description": (
                "Isaac world to MuJoCo G1 using the authored G1 world yaw and "
                "the USD-root/MuJoCo floating-base z reference."
            ),
            "robot_rotation_world": frames["robot_rotation_world"],
            "robot_translation_world_from_mujoco": frames[
                "robot_translation_world_from_mujoco"],
        },
        "initial_phone_pose": {
            "dimensions_xyz_m": phone_dims,
            "center_world_m": frames["phone_center_world"],
            "rotation_world": frames["phone_rotation_world"],
            "center_g1_m": frames["phone_center_g1"],
            "rotation_g1": frames["phone_rotation_g1"],
            "world_axes": frames["phone_axes_world"],
            "screen_normal_definition": "phone local -Y",
            "dynamic_scene_lean_deg_about_world_x": magnet[
                "initial_assembly"]["lean_degrees_about_world_x"],
        },
        "world_axes": {
            "table_normal": frames["table_normal_world"],
            "torso_forward": frames["robot_rotation_world"][:, 0],
            "torso_lateral": frames["robot_rotation_world"][:, 1],
            "torso_up": frames["robot_rotation_world"][:, 2],
        },
        "charger_pose": {
            "face_center_world_m": frames["charger_face_center_world"],
            "charging_normal_world": frames["charger_normal_world"],
            "target_rotation_world": frames["charger_rotation_world"],
            "face_center_g1_m": frames["charger_center_g1"],
            "target_rotation_g1": frames["charger_rotation_g1"],
        },
        "automatic_phase_detection": {
            "method": (
                "Schmitt hysteresis on both ALOHA grippers; rank contiguous "
                "bilateral-closed runs by measured ALOHA midpoint lift; derive "
                "lift-clearance crossing from that run."
            ),
            "candidate_runs": phase["candidates"],
            "selected_run": phase["selected"],
            "boundaries": phase["boundaries"],
            "phase_counts": {
                label: int(np.sum(phase["labels"] == label))
                for label in np.unique(phase["labels"])
            },
        },
        "planned_interpolation": {
            "contact_orientation": "actual initial phone world orientation",
            "lift_orientation": "actual initial phone world orientation",
            "carry_rotation": "SO(3) geodesic with minimum-jerk scalar",
            "place_rotation": "SO(3) geodesic toward charger target rotation",
            "minimum_lift_clearance_rule": (
                "max(10 mm, half automatically observed lift), capped at 30 mm"
            ),
            "minimum_jerk_definition": "10u^3-15u^4+6u^5",
        },
        "joint_margin_trials": audits,
        "selected_joint_margin_rad": selected_margin,
        "phone_nearest_forward_coordinate_g1_m": nearest,
        "position_gate_removed": True,
        "source_position_costs_intended": [
            "midpoint deviation", "relative-vector deviation",
            "original left/right hand-position deviation",
        ],
        "unavailable_due_to_contact_conflict": [
            "phone target pose trajectory", "contact frames",
            "lift clearance trajectory", "screen/torso angle trajectory",
            "phone/charger relative pose trajectory",
            "joint-limit margin trajectory", "full G1 qpos",
            "collision statistics", "kinematic videos",
        ],
        "isaac_lab_executed": False, "hardware_executed": False,
        "mujoco_mode": "read-only FK/reach-envelope kinematic audit",
    }
    report = serial(report)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    if a.execute:
        if TRAJECTORY.exists():
            raise FileExistsError(
                f"Refusing to overwrite unverified trajectory: {TRAJECTORY}")
        temp = a.report.with_suffix(".json.incomplete")
        temp.write_text(json.dumps(report, indent=2))
        os.replace(temp, a.report)
    print(json.dumps(report, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
