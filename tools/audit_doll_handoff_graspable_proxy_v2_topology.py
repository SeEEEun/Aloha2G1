#!/usr/bin/env python3
"""Kinematic table/object-clearance audit for the proxy-v2 power grasp.

This utility is deliberately offline: qpos assignment plus MuJoCo forward
kinematics and exact geom-distance queries only.  It never imports a learned
policy, steps physics, or writes a robot command.  It is used to reject palm
orientations that would put Dex3 collision geometry through the table before
spending an Isaac calibration trial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import find_g1_dex3_static_phone_grasp as old  # noqa: E402
import refine_g1_dex3_static_phone_contact as contact  # noqa: E402
from tools.doll_handoff_retargeting.common import (  # noqa: E402
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import authoritative_joint_ranges  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/doll_handoff_graspable_proxy_v2.json",
    )
    parser.add_argument("--geometry", default="G1_60x55x50")
    parser.add_argument("--tilts-deg", default="-90,-75,-60,-45,-30,0,30,45,60,75,90")
    parser.add_argument("--center-x-offsets-mm", default="0")
    parser.add_argument("--center-y-offsets-mm", default="0")
    parser.add_argument("--center-z-offsets-mm", default="0,5,10,15,20,25,30")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def hand_in_model_order(
    g1: G1Kinematics,
    canonical_names: list[str],
    values: list[float],
    side: str,
) -> np.ndarray:
    block = slice(14, 21) if side == "left" else slice(21, 28)
    by_name = dict(zip(canonical_names[block], values, strict=True))
    return np.asarray([by_name[name] for name in g1.hand_joint_names[side]])


def expanded_model(
    g1: G1Kinematics,
    object_center_model: np.ndarray,
    dimensions: np.ndarray,
    table_z_model: float,
) -> mujoco.MjModel:
    base = mujoco.MjModel.from_xml_path(str(g1.path))
    directory = Path(tempfile.mkdtemp(prefix="proxy_v2_topology_"))
    xml_path = directory / "model.xml"
    mujoco.mj_saveLastXML(str(xml_path), base)
    text = xml_path.read_text(encoding="utf-8")
    text = text.replace(
        'meshdir="assets/"', f'meshdir="{g1.path.parent / "assets"}/"'
    )
    center = " ".join(f"{value:.12g}" for value in object_center_model)
    radii = " ".join(f"{value:.12g}" for value in dimensions / 2.0)
    extra = (
        f'<body name="proxy_v2" pos="{center}">'
        f'<geom name="proxy_v2_geom" type="ellipsoid" size="{radii}" '
        'contype="1" conaffinity="1"/>'
        '</body>'
        f'<body name="proxy_v2_table" pos="0 0 {table_z_model - 0.05:.12g}">'
        '<geom name="proxy_v2_table_geom" type="box" size="2 2 .05" '
        'contype="1" conaffinity="1"/>'
        '</body>'
    )
    text = text.replace("<worldbody>", "<worldbody>\n    " + extra, 1)
    xml_path.write_text(text, encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(xml_path))


def exact_distance(
    model: mujoco.MjModel, data: mujoco.MjData, first: int, second: int
) -> float:
    """Preserve real closest-point separation when MuJoCo reports false zero."""

    value, from_to = contact.distance(model, data, first, second)
    closest = float(np.linalg.norm(from_to[3:] - from_to[:3]))
    return closest if value == 0.0 and closest > 1.0e-12 else value


def main() -> int:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    geometry = next(
        row for row in config["geometry_candidates"] if row["name"] == args.geometry
    )
    dimensions = np.asarray(geometry["dimensions_m"], dtype=np.float64)
    canonical_names, _ = authoritative_joint_ranges()
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal, _ = g1.derive_task_ready_nominal(common, scene)
    side = args.side
    other = "right" if side == "left" else "left"
    selected_power = hand_in_model_order(
        g1,
        canonical_names,
        config["hand_states"][side]["POWER_GRASP_P1"],
        side,
    )
    open_hand = hand_in_model_order(
        g1, canonical_names, config["hand_states"][side]["OPEN"], side
    )
    preshape = hand_in_model_order(
        g1, canonical_names, config["hand_states"][side]["PRESHAPE"], side
    )
    other_open = hand_in_model_order(
        g1, canonical_names, config["hand_states"][other]["OPEN"], other
    )
    if side == "left":
        g1.assign(nominal, selected_power, other_open)
    else:
        g1.assign(nominal, other_open, selected_power)
    wrist_to_tool = np.linalg.inv(g1.wrist_pose(side)) @ g1.whole_hand_grasp_pose(side)
    origin_world = g1.model_to_world_position(np.zeros(3, dtype=np.float64))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3, dtype=np.float64)[axis])
            - origin_world
            for axis in range(3)
        ]
    )
    current_world = world_from_model @ g1.whole_hand_grasp_pose(side)[:3, :3]
    closing = current_world[:, 0].copy()
    closing[2] = 0.0
    closing /= np.linalg.norm(closing)
    spread = current_world[:, 2].copy()
    spread[2] = 0.0
    spread -= closing * float(spread @ closing)
    spread /= np.linalg.norm(spread)
    transverse = np.cross(spread, closing)
    horizontal_frame = np.column_stack((closing, transverse, spread))
    object_center_world = np.asarray(
        [
            *config["object"]["center_world_xy_m"],
            float(config["object"]["table_surface_world_z_m"])
            + dimensions[2] / 2.0,
        ],
        dtype=np.float64,
    )
    object_center_model = g1.world_to_model_position(object_center_world)
    table_point_model = g1.world_to_model_position(
        np.asarray([*config["object"]["center_world_xy_m"], config["object"]["table_surface_world_z_m"]])
    )
    model = expanded_model(g1, object_center_model, dimensions, float(table_point_model[2]))
    data = mujoco.MjData(model)
    proxy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "proxy_v2_geom")
    table = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "proxy_v2_table_geom"
    )
    arm_qpos = np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in g1.arm_joint_names
        ],
        dtype=np.int64,
    )
    hand_qpos = np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in g1.hand_joint_names[side]
        ],
        dtype=np.int64,
    )
    active_geoms: list[tuple[int, str]] = []
    for geom_id in range(model.ngeom):
        body = old.body_name(model, geom_id)
        if (
            body.startswith(f"{side}_hand") or body.startswith(f"{side}_wrist")
        ) and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            active_geoms.append((geom_id, body))
    distal: dict[str, int] = {}
    for digit in ("thumb", "index", "middle"):
        suffix = "2_link" if digit == "thumb" else "1_link"
        distal[digit] = contact.collision_geoms(
            model, f"{side}_hand_{digit}_{suffix}"
        )[-1]

    def evaluate(arm: np.ndarray, hand: np.ndarray) -> dict[str, object]:
        data.qpos[:] = model.key_qpos[0]
        data.qpos[arm_qpos] = arm
        data.qpos[hand_qpos] = hand
        mujoco.mj_forward(model, data)
        object_rows = [
            (exact_distance(model, data, geom_id, proxy), body)
            for geom_id, body in active_geoms
        ]
        table_rows = [
            (exact_distance(model, data, geom_id, table), body)
            for geom_id, body in active_geoms
        ]
        return {
            "minimum_object_clearance_m": min(object_rows)[0],
            "minimum_object_clearance_body": min(object_rows)[1],
            "minimum_table_clearance_m": min(table_rows)[0],
            "minimum_table_clearance_body": min(table_rows)[1],
            "distal_object_distances_m": {
                digit: exact_distance(model, data, geom_id, proxy)
                for digit, geom_id in distal.items()
            },
        }

    block = slice(0, 7) if side == "left" else slice(7, 14)
    lower = g1.arm_limits[block, 0] + 1.0e-6
    upper = g1.arm_limits[block, 1] - 1.0e-6
    seed = nominal[block].copy()
    tilts = [float(value) for value in args.tilts_deg.split(",")]
    x_offsets = [float(value) for value in args.center_x_offsets_mm.split(",")]
    y_offsets = [float(value) for value in args.center_y_offsets_mm.split(",")]
    z_offsets = [float(value) for value in args.center_z_offsets_mm.split(",")]
    rows: list[dict[str, object]] = []
    for tilt_deg in tilts:
        tilt = np.deg2rad(tilt_deg)
        rotation_about_closing = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(tilt), -np.sin(tilt)],
                [0.0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        desired_world_rotation = horizontal_frame @ rotation_about_closing
        desired_model_rotation = world_from_model.T @ desired_world_rotation
        for x_offset_mm in x_offsets:
          for y_offset_mm in y_offsets:
           for z_offset_mm in z_offsets:
            target_world = object_center_world + np.asarray(
                [x_offset_mm, y_offset_mm, z_offset_mm], dtype=np.float64
            ) / 1000.0
            target_model = g1.world_to_model_position(target_world)

            def residual(active: np.ndarray) -> np.ndarray:
                full = nominal.copy()
                full[block] = active
                if side == "left":
                    g1.assign(full, selected_power, other_open)
                else:
                    g1.assign(full, other_open, selected_power)
                position, rotation, _, _ = g1.static_tool_pose_state(side, wrist_to_tool)
                return np.r_[
                    100.0 * (position - target_model),
                    8.0
                    * Rotation.from_matrix(rotation.T @ desired_model_rotation).as_rotvec(),
                    0.01 * (active - seed),
                ]

            solution = least_squares(
                residual,
                np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=600,
                ftol=1.0e-11,
                xtol=1.0e-11,
                gtol=1.0e-11,
            )
            arm = nominal.copy()
            arm[block] = solution.x
            if side == "left":
                g1.assign(arm, selected_power, other_open)
            else:
                g1.assign(arm, other_open, selected_power)
            achieved_position, achieved_rotation, _, _ = g1.static_tool_pose_state(
                side, wrist_to_tool
            )
            open_audit = evaluate(arm, open_hand)
            preshape_audit = evaluate(arm, preshape)
            power_candidates: list[dict[str, object]] = []
            for alpha in np.linspace(0.0, 1.0, 41):
                hand = preshape + alpha * (selected_power - preshape)
                audit = evaluate(arm, hand)
                distal_values = np.asarray(
                    list(audit["distal_object_distances_m"].values())
                )
                audit["alpha_preshape_to_p1"] = float(alpha)
                audit["contact_balance_score"] = float(
                    np.sum((distal_values + 0.0005) ** 2)
                    + 1000.0 * max(0.0, 0.001 - audit["minimum_table_clearance_m"]) ** 2
                )
                power_candidates.append(audit)
            best_power = min(
                power_candidates, key=lambda row: row["contact_balance_score"]
            )
            rows.append(
                {
                    "tilt_deg": tilt_deg,
                    "center_x_offset_mm": x_offset_mm,
                    "center_y_offset_mm": y_offset_mm,
                    "center_z_offset_mm": z_offset_mm,
                    "arm_q_rad": arm[block].tolist(),
                    "ik_position_error_m": float(
                        np.linalg.norm(achieved_position - target_model)
                    ),
                    "ik_orientation_error_rad": float(
                        np.linalg.norm(
                            Rotation.from_matrix(
                                achieved_rotation.T @ desired_model_rotation
                            ).as_rotvec()
                        )
                    ),
                    "open": open_audit,
                    "preshape": preshape_audit,
                    "best_power_on_preshape_to_p1": best_power,
                }
            )
    payload = {
        "schema_version": "doll_handoff_graspable_proxy_v2_kinematic_topology_audit",
        "mode": "MuJoCo qpos plus forward kinematics and geom distance only",
        "policy_used": False,
        "side": side,
        "geometry": geometry,
        "tilts_deg": tilts,
        "center_x_offsets_mm": x_offsets,
        "center_y_offsets_mm": y_offsets,
        "center_z_offsets_mm": z_offsets,
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        best = row["best_power_on_preshape_to_p1"]
        distances = best["distal_object_distances_m"]
        print(
            f"tilt={row['tilt_deg']:>5.1f} "
            f"xyz={row['center_x_offset_mm']:>4.0f}/"
            f"{row['center_y_offset_mm']:>4.0f}/"
            f"{row['center_z_offset_mm']:>4.0f}mm "
            f"IK={row['ik_position_error_m']*1000:5.1f}mm/"
            f"{np.rad2deg(row['ik_orientation_error_rad']):4.1f}deg "
            f"O/P-table={row['open']['minimum_table_clearance_m']*1000:6.1f}/"
            f"{row['preshape']['minimum_table_clearance_m']*1000:6.1f}mm "
            f"alpha={best['alpha_preshape_to_p1']:.2f} "
            f"power-table={best['minimum_table_clearance_m']*1000:6.1f}mm "
            f"T/I/M={distances['thumb']*1000:6.1f}/"
            f"{distances['index']*1000:6.1f}/{distances['middle']*1000:6.1f}mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
