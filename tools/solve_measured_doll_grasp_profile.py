#!/usr/bin/env python3
"""Solve one bounded mirrored Dex3 profile for a predeclared v3 palm pose.

This is an offline geometry utility.  It uses named joints, MuJoCo forward
kinematics, and exact distance queries only; it never imports a policy, starts
Isaac, or writes a command to hardware.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import find_g1_dex3_static_phone_grasp as old  # noqa: E402
import refine_g1_dex3_static_phone_contact as contact  # noqa: E402
from tools.audit_doll_handoff_graspable_proxy_v2_topology import (  # noqa: E402
    exact_distance,
    expanded_model,
    hand_in_model_order,
)
from tools.doll_handoff_retargeting.common import (  # noqa: E402
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import authoritative_joint_ranges  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry", default="MEASURED_100")
    parser.add_argument("--left-audit", type=Path, required=True)
    parser.add_argument("--right-audit", type=Path, required=True)
    parser.add_argument("--left-x-mm", type=float, required=True)
    parser.add_argument("--right-x-mm", type=float, required=True)
    parser.add_argument("--y-mm", type=float, required=True)
    parser.add_argument("--z-mm", type=float, required=True)
    parser.add_argument("--left-tilt-deg", type=float, required=True)
    parser.add_argument("--right-tilt-deg", type=float, required=True)
    parser.add_argument("--preshape-fraction", type=float, default=0.15)
    parser.add_argument("--target-penetration-mm", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_arm_row(
    path: Path,
    *,
    tilt_deg: float,
    x_mm: float,
    y_mm: float,
    z_mm: float,
) -> dict[str, Any]:
    rows = read_json(path)["rows"]
    matches = [
        row
        for row in rows
        if np.isclose(row["tilt_deg"], tilt_deg)
        and np.isclose(row["center_x_offset_mm"], x_mm)
        and np.isclose(row["center_y_offset_mm"], y_mm)
        and np.isclose(row["center_z_offset_mm"], z_mm)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one matching pose in {path}, got {len(matches)}")
    return matches[0]


def main() -> int:
    args = arguments()
    config = read_json(args.config)
    if config.get("schema_version") != "doll_handoff_measured_proxy_v3":
        raise RuntimeError("this solver is restricted to measured proxy v3")
    if not 0.0 < args.preshape_fraction < 1.0:
        raise ValueError("preshape fraction must be strictly between zero and one")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal, _ = g1.derive_task_ready_nominal(common, scene)
    names, _ = authoritative_joint_ranges()
    geometry = next(
        row for row in config["geometry_candidates"] if row["name"] == args.geometry
    )
    dimensions = np.asarray(geometry["dimensions_m"], dtype=np.float64)
    center_world = np.asarray(
        [
            *config["object"]["center_world_xy_m"],
            float(config["object"]["table_surface_world_z_m"]) + dimensions[2] / 2.0,
        ],
        dtype=np.float64,
    )
    table_world = np.asarray(
        [
            *config["object"]["center_world_xy_m"],
            float(config["object"]["table_surface_world_z_m"]),
        ],
        dtype=np.float64,
    )
    model = expanded_model(
        g1,
        g1.world_to_model_position(center_world),
        dimensions,
        float(g1.world_to_model_position(table_world)[2]),
    )
    data = mujoco.MjData(model)
    proxy_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "proxy_v2_geom"
    )
    table_id = mujoco.mj_name2id(
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
    hand_qpos = {
        side: np.asarray(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in g1.hand_joint_names[side]
            ],
            dtype=np.int64,
        )
        for side in ("left", "right")
    }
    active_geoms: dict[str, list[tuple[int, str]]] = {}
    distal_geoms: dict[str, dict[str, int]] = {}
    for side in ("left", "right"):
        active_geoms[side] = [
            (geom_id, old.body_name(model, geom_id))
            for geom_id in range(model.ngeom)
            if (
                old.body_name(model, geom_id).startswith(f"{side}_hand")
                or old.body_name(model, geom_id).startswith(f"{side}_wrist")
            )
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ]
        distal_geoms[side] = {
            digit: contact.collision_geoms(
                model,
                f"{side}_hand_{digit}_{'2_link' if digit == 'thumb' else '1_link'}",
            )[-1]
            for digit in ("thumb", "index", "middle")
        }

    pose_rows = {
        "left": select_arm_row(
            args.left_audit,
            tilt_deg=args.left_tilt_deg,
            x_mm=args.left_x_mm,
            y_mm=args.y_mm,
            z_mm=args.z_mm,
        ),
        "right": select_arm_row(
            args.right_audit,
            tilt_deg=args.right_tilt_deg,
            x_mm=args.right_x_mm,
            y_mm=args.y_mm,
            z_mm=args.z_mm,
        ),
    }
    arms: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        arms[side] = nominal.copy()
        block = slice(0, 7) if side == "left" else slice(7, 14)
        arms[side][block] = np.asarray(pose_rows[side]["arm_q_rad"])

    mirror = np.asarray([1, -1, -1, -1, -1, -1, -1], dtype=np.float64)

    def audit(left_canonical: np.ndarray) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for side, canonical in (
            ("left", left_canonical),
            ("right", left_canonical * mirror),
        ):
            hand = hand_in_model_order(g1, names, canonical.tolist(), side)
            data.qpos[:] = model.key_qpos[0]
            data.qpos[arm_qpos] = arms[side]
            data.qpos[hand_qpos[side]] = hand
            mujoco.mj_forward(model, data)
            object_rows = [
                (exact_distance(model, data, geom, proxy_id), body)
                for geom, body in active_geoms[side]
            ]
            table_rows = [
                (exact_distance(model, data, geom, table_id), body)
                for geom, body in active_geoms[side]
            ]
            output[side] = {
                "minimum_object": min(object_rows),
                "minimum_table": min(table_rows),
                "distal": np.asarray(
                    [
                        exact_distance(
                            model, data, distal_geoms[side][digit], proxy_id
                        )
                        for digit in ("thumb", "index", "middle")
                    ]
                ),
            }
        return output

    open_left = np.asarray(config["hand_states"]["left"]["OPEN"], dtype=np.float64)
    lower = np.asarray([-0.6, 0.4, 0.0, -0.8, -1.5, -0.8, -1.5])
    upper = np.asarray([0.6, 1.047, 1.745, 0.0, 0.0, 0.0, 0.0])
    seeds = [
        np.asarray([0.1, 0.75, 0.3, -0.15, -0.5, -0.15, -0.4]),
        np.asarray([0.2, 1.0, 0.7, -0.1, -0.7, -0.1, -0.6]),
        np.asarray([-0.2, 0.95, 1.1, -0.1, -0.8, -0.1, -0.7]),
    ]
    target = -float(args.target_penetration_mm) / 1000.0
    candidates: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        def residual(value: np.ndarray) -> np.ndarray:
            state = audit(value)
            distances = np.concatenate(
                [state[side]["distal"] for side in ("left", "right")]
            )
            penetration_penalty = np.asarray(
                [
                    2000.0
                    * max(
                        0.0,
                        -0.0025 - float(state[side]["minimum_object"][0]),
                    )
                    for side in ("left", "right")
                ]
            )
            return np.r_[
                1000.0 * (distances - target),
                penetration_penalty,
                0.02 * (value - seed),
            ]

        solution = least_squares(
            residual,
            np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=700,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        state = audit(solution.x)
        preshape = open_left + args.preshape_fraction * (solution.x - open_left)
        preshape_state = audit(preshape)
        distal = np.concatenate(
            [state[side]["distal"] for side in ("left", "right")]
        )
        candidates.append(
            {
                "seed_index": seed_index,
                "solution_left_7d": solution.x.tolist(),
                "solution_right_7d": (solution.x * mirror).tolist(),
                "preshape_left_7d": preshape.tolist(),
                "preshape_right_7d": (preshape * mirror).tolist(),
                "cost": float(solution.cost),
                "maximum_absolute_distal_target_error_m": float(
                    np.max(np.abs(distal - target))
                ),
                "sides": {
                    side: {
                        "distal_object_distances_m": state[side]["distal"].tolist(),
                        "minimum_object_clearance_m": float(
                            state[side]["minimum_object"][0]
                        ),
                        "minimum_object_clearance_body": state[side]["minimum_object"][1],
                        "minimum_table_clearance_m": float(
                            state[side]["minimum_table"][0]
                        ),
                        "minimum_table_clearance_body": state[side]["minimum_table"][1],
                        "preshape_minimum_object_clearance_m": float(
                            preshape_state[side]["minimum_object"][0]
                        ),
                        "preshape_minimum_table_clearance_m": float(
                            preshape_state[side]["minimum_table"][0]
                        ),
                    }
                    for side in ("left", "right")
                },
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["maximum_absolute_distal_target_error_m"], row["cost"]
        ),
    )
    payload = {
        "schema_version": "doll_handoff_measured_proxy_v3_grasp_profile_design",
        "mode": "offline named-joint FK and exact distance only",
        "policy_used": False,
        "real_robot_used": False,
        "config": str(args.config.resolve()),
        "geometry_dimensions_m": dimensions.tolist(),
        "geometry": geometry,
        "pose": {
            "left_x_mm": args.left_x_mm,
            "right_x_mm": args.right_x_mm,
            "y_mm": args.y_mm,
            "z_mm": args.z_mm,
            "left_tilt_deg": args.left_tilt_deg,
            "right_tilt_deg": args.right_tilt_deg,
        },
        "target_penetration_m": target,
        "preshape_fraction": args.preshape_fraction,
        "candidates": candidates,
        "selected": selected,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
