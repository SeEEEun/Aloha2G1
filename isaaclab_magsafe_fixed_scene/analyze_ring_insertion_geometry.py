#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "converted_runs/magsafe_20260724_154440/isaac_replay"
paths = list(OUT.glob("right_ring_geometry_raw_*ring_insertion_audit_v1.csv"))
if len(paths) != 1:
    raise RuntimeError(paths)
with paths[0].open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
approach = [r for r in rows if 480 <= int(r["source_frame"]) <= 630]

def a(side, field):
    return np.array([float(r[f"{side}_{field}"]) for r in approach])

stats = {}
for side in ("left", "right"):
    radial = a(side, "center_radial_offset_m")
    i = int(np.argmin(radial))
    r = approach[i]
    stats[side] = {
        "best_frame": int(r["source_frame"]),
        "tip_ring_xyz_m": [float(r[f"{side}_tip_ring_{c}"]) for c in "xyz"],
        "center_offset_m": float(radial[i]),
        "approach_angle_error_deg": float(r[f"{side}_axis_normal_angle_deg"]),
        "nearest_inner_edge_clearance_m": float(r[f"{side}_inner_edge_clearance_m"]),
        "minimum_plane_distance_m": float(np.min(a(side, "tip_ring_z"))),
        "radial_range_m": [float(np.min(radial)), float(np.max(radial))],
        "plane_distance_range_m": [
            float(np.min(a(side, "tip_ring_z"))), float(np.max(a(side, "tip_ring_z")))
        ],
        "angle_range_deg": [
            float(np.min(a(side, "axis_normal_angle_deg"))),
            float(np.max(a(side, "axis_normal_angle_deg"))),
        ],
    }

selected = "left"
s = stats[selected]
correction = [-x for x in s["tip_ring_xyz_m"]]
correction_norm = float(np.linalg.norm(correction))

timeline_path = OUT / "right_finger_ring_relative_timeline.csv"
with timeline_path.open("x", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

(OUT / "accessory_ring_frame_report.txt").open("x", encoding="utf-8").write(
    "source=/World/MagSafeScene/Accessory/Colliders/MainRing/Segment_00..11\n"
    "origin=main C-ring opening geometric center at accessory local [0,0,0]\n"
    "ring x=accessory local +X\nring y=accessory local -Z\n"
    "ring z=accessory local +Y (phone rear outward normal)\n"
    "inner_radius_m=0.0225\nouter_radius_m=0.0275\n"
    "radial_ring_width_m=0.005\naxial_depth_m=0.0035\n"
    "gap_degrees=36\ngap_center_degrees=-90\n"
    "T_accessory_ring=translation identity; rotation maps [X,-Z,+Y]\n"
)

(OUT / "right_ring_insertion_geometry_report.txt").open("x", encoding="utf-8").write(
    json.dumps({
        "candidate_fingers": stats,
        "selected_insertion_finger": selected,
        "insertion_finger_cross_section_m": [0.00497883, 0.01965619],
        "inner_diameter_m": .045,
        "RING_INSERTION_PASS": False,
        "reason": "tip never crosses ring plane; side contact occurs first",
    }, indent=2) + "\n"
)

(OUT / "right_open_loop_pose_error_report.txt").open("x", encoding="utf-8").write(
    json.dumps({
        "classification": "OPEN_LOOP_WORLD_TRAJECTORY_NOT_POSE_INVARIANT",
        "left_tip_best": s,
        "error_is_constant_rigid_offset": False,
        "radial_error_variation_m": s["radial_range_m"],
        "normal_error_variation_m": s["plane_distance_range_m"],
        "approach_angle_variation_deg": s["angle_range_deg"],
    }, indent=2) + "\n"
)

(OUT / "right_fixed_scene_calibration_report.txt").open("x", encoding="utf-8").write(
    json.dumps({
        "best_frame_ring_relative_translation_to_center_m": correction,
        "translation_norm_m": correction_norm,
        "required_orientation_correction_deg": s["approach_angle_error_deg"],
        "allowed_phone_initial_xy_m": .005,
        "allowed_phone_initial_yaw_deg": 5,
        "feasible_with_allowed_fixed_calibration": False,
        "result": "NOT_RUN_GEOMETRICALLY_INFEASIBLE",
    }, indent=2) + "\n"
)

(OUT / "right_ring_relative_retargeting_report.txt").open("x", encoding="utf-8").write(
    json.dumps({
        "required": True,
        "pose_provider_interface": [
            "simulation_ground_truth", "perception_estimate", "fixed_scene_calibration"
        ],
        "selected_provider_for_isaac": "simulation_ground_truth",
        "target_equation": "T_world_hand_target=T_world_ring_current*T_ring_hand_demo",
        "demonstration_relative_trajectory_available": False,
        "blocker": "dataset contains joint states and RGB but no calibrated 3-D ring/hand demonstration poses",
        "IK_success_rate": None,
        "physical_result": "NOT_RUN",
        "hardcoded_world_coordinates": False,
    }, indent=2) + "\n"
)

(OUT / "right_accessory_method_comparison.txt").open("x", encoding="utf-8").write(
    "method,result,insertion,bilateral,detach\n"
    "open_loop,FAIL,FAIL,FAIL,FAIL\n"
    "fixed_scene_calibration,GEOMETRICALLY_INFEASIBLE,NOT_RUN,NOT_RUN,NOT_RUN\n"
    "ring_relative_retargeting,REQUIRED_BUT_DEMO_3D_POSE_UNAVAILABLE,NOT_RUN,NOT_RUN,NOT_RUN\n"
)

(OUT / "right_accessory_task_definition.txt").open("x", encoding="utf-8").write(
    "1 pre-insertion: insertion finger outside ring plane and centered in opening\n"
    "2 insertion: finger cross-section passes through inner radius with positive clearance\n"
    "3 grasp: insertion finger inside, opposing finger outside, bilateral edge contact\n"
    "4 pull: move along ring +Z (phone rear outward normal)\n"
    "5 detach: configured physical right-contact wrench threshold only\n"
)

keydir = OUT / "right_ring_insertion_keyframes"
keydir.mkdir(exist_ok=True)
best = min(approach, key=lambda r: float(r["left_center_radial_offset_m"]))
for view in ("ring_normal", "side", "front"):
    fig, ax = plt.subplots(figsize=(6, 5))
    if view == "ring_normal":
        circle = plt.Circle((0, 0), .0225, fill=False, color="black", lw=2)
        ax.add_patch(circle)
        for side, color in (("left", "red"), ("right", "blue")):
            x, y = [float(best[f"{side}_tip_ring_{c}"]) for c in "xy"]
            ax.scatter([x], [y], c=color, label=f"{side} tip")
        ax.set(xlabel="ring X (m)", ylabel="ring Y (m)", aspect="equal")
    elif view == "side":
        for side, color in (("left", "red"), ("right", "blue")):
            y, z = [float(best[f"{side}_tip_ring_{c}"]) for c in "yz"]
            ax.scatter([z], [y], c=color, label=f"{side} tip")
        ax.axvline(0, color="black", label="ring plane")
        ax.set(xlabel="ring-normal Z (m)", ylabel="ring Y (m)")
    else:
        for side, color in (("left", "red"), ("right", "blue")):
            x, z = [float(best[f"{side}_tip_ring_{c}"]) for c in ("x", "z")]
            ax.scatter([z], [x], c=color, label=f"{side} tip")
        ax.axvline(0, color="black", label="ring plane")
        ax.set(xlabel="ring-normal Z (m)", ylabel="ring X (m)")
    ax.legend()
    ax.grid(True)
    ax.set_title(f"Isaac ring insertion audit frame {best['source_frame']}")
    fig.tight_layout()
    fig.savefig(keydir / f"{view}_frame_{int(best['source_frame']):06d}.png", dpi=160)
    plt.close(fig)

print(json.dumps({"selected": selected, "stats": s, "correction": correction}, indent=2))
