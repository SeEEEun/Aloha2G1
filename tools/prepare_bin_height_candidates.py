#!/usr/bin/env python3
"""Predeclare the bounded geometry-driven bin-height calibration candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/final_bin_calibrated_completion/01_selected_bin"
CONTACT = (
    ROOT
    / "outputs/final_bin_calibrated_completion/00_bin_collision_audit"
    / "physics_exact_replay/robot_bin_contacts.npz"
)
SCENE = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
CANDIDATE_HEIGHTS_M = (0.190, 0.140, 0.120, 0.115, 0.105, 0.100)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main() -> int:
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    with np.load(CONTACT, allow_pickle=False) as archive:
        force = np.asarray(archive["force_n"], dtype=np.float64)
        points = np.asarray(archive["point_world_m"], dtype=np.float64)
    physical = force > 1e-6
    lowest_observed_collision_z = float(np.min(points[physical, 2]))
    table_z = float(scene["table"]["surface_height_m"])
    doll_visual_height = 0.085
    minimum_height = max(0.100, 1.15 * doll_visual_height)
    rows = []
    for height in CANDIDATE_HEIGHTS_M:
        rim_z = table_z + height
        clearance = lowest_observed_collision_z - rim_z
        rows.append(
            {
                "height_m": height,
                "rim_world_z_m": rim_z,
                "minimum_height_gate_m": minimum_height,
                "capacity_gate": "PASS" if height >= minimum_height else "FAIL",
                "opening_unchanged": True,
                "bin_xy_unchanged": True,
                "wall_thickness_unchanged": True,
                "ik_reachability": "UNCHANGED_BY_HEIGHT_ONLY_EDIT",
                "conservative_clearance_estimate_m": clearance,
                "offline_collision_prediction": (
                    "SAFE_CANDIDATE" if clearance >= 0.003 else "REJECT_OR_BOUNDARY"
                ),
            }
        )
    payload = {
        "schema_version": "bounded_bin_height_candidates_v1",
        "policy_independent": True,
        "selected_before_act_ab": True,
        "selection_rule": "highest physically reasonable candidate passing all right-only physical gates",
        "source_scene": str(SCENE),
        "source_scene_sha256": sha256(SCENE),
        "source_contact_trace": str(CONTACT),
        "source_contact_trace_sha256": sha256(CONTACT),
        "original_height_m": float(scene["bin"]["outer_dimensions_xyz_m"][2]),
        "reasonable_minimum_height_m": minimum_height,
        "lowest_observed_right_elbow_bin_contact_z_m": lowest_observed_collision_z,
        "conservative_clearance_margin_m": 0.003,
        "candidates": rows,
        "physics_test_order": [0.120, 0.115, 0.105, 0.100],
        "physics_early_exit": "stop at highest candidate passing all gates",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "BIN_HEIGHT_CANDIDATES.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=default) + "\n")
    table = [
        "# Bounded Bin Height Candidates",
        "",
        "Selection is policy-independent: choose the highest candidate passing all right-only physical gates.",
        "",
        "| Height (mm) | Rim Z (m) | Estimated clearance (mm) | Capacity | Offline result |",
        "|---:|---:|---:|:---:|:---|",
    ]
    for row in rows:
        table.append(
            f"| {1000*row['height_m']:.0f} | {row['rim_world_z_m']:.3f} | "
            f"{1000*row['conservative_clearance_estimate_m']:.3f} | {row['capacity_gate']} | "
            f"{row['offline_collision_prediction']} |"
        )
    table += [
        "",
        f"The measured collision sweep begins at Z={lowest_observed_collision_z:.6f} m. "
        "The 120 mm candidate is the boundary test; 115 mm is the first candidate with at least 3 mm estimated clearance.",
        "",
        "The opening footprint, bin XY pose, wall thickness, bottom, material, robot, doll, controller, and safety thresholds remain unchanged.",
    ]
    (OUTPUT / "BIN_HEIGHT_CANDIDATES.md").write_text("\n".join(table) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
