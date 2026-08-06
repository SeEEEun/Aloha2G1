#!/usr/bin/env python3
"""Analyze adaptive effort-cap threshold and lifted-state pivot candidates."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "converted_runs/magsafe_20260724_154440/isaac_replay"
REPORT_DIR = OUT / "phone_intermediate_effort_candidate_reports"
REPORT_DIR.mkdir(exist_ok=True)
HALF = np.array([.1496, .00795, .0715]) / 2
TABLE_Z = .795
MASS = .217
WEIGHT = MASS * 9.81
G = np.array([0., 0., -WEIGHT])
COM_LOCAL = np.array([.0044, .0024758, -.0017408])
SOURCE_SPEED = .25
REQUIRED_SIM_DURATION = .2 / SOURCE_SPEED

SEARCH = [
    ("S1", 11.291945, "effort_search_S1"),
    ("S2", 6.121925, "effort_search_S2"),
    ("S3", 4.507623, "effort_search_S3"),
    ("S4", 5.253126, "effort_search_S4"),
    ("S5", 4.866119, "effort_search_S5"),
]
WINDOW = [
    ("W1", 4.866119, "pivot_window_W1"),
    ("W2", 5.109425, "pivot_window_W2"),
    ("W3", 5.352731, "pivot_window_W3"),
    ("W4", 5.839343, "pivot_window_W4"),
]


def rows(pattern):
    paths = list(OUT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"{pattern}: expected one file, found {paths}")
    with paths[0].open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rot(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pct(v):
    return [float(x) for x in np.percentile(np.asarray(v, float), [5, 50, 95])] if v else [math.nan]*3


def duration(records, predicate):
    best = cur = 0.
    previous = None
    for r in records:
        t = r["time"]
        dt = 0. if previous is None else t - previous
        cur = cur + dt if predicate(r) else 0.
        best = max(best, cur)
        previous = t
    return best


def base_motion(tag):
    data = rows(f"phone_passive_rotation_*{tag}.csv")
    out = []
    for r in data:
        frame = int(r["source_frame"])
        if frame < 280:
            continue
        rp = rot([float(r[k]) for k in ("quat_x", "quat_y", "quat_z", "quat_w")])
        rw = rot([float(r[k]) for k in ("wrist_quat_x", "wrist_quat_y", "wrist_quat_z", "wrist_quat_w")])
        phone = np.array([float(r[k]) for k in ("phone_x", "phone_y", "phone_z")])
        out.append({
            "time": float(r["sim_time"]), "frame": frame, "rp": rp, "rw": rw, "phone": phone,
            "clearance": phone[2] - np.abs(rp[2]) @ HALF - TABLE_Z,
            "portrait": float(r["long_axis_vertical_error_deg"]),
            "omega": np.array([float(r[k]) for k in ("omega_x", "omega_y", "omega_z")]),
            "pinch": np.array([float(r[k]) for k in ("pinch_axis_x", "pinch_axis_y", "pinch_axis_z")]),
            "contact_count": int(r["contact_count"]),
            "long_gripper": rw.T @ rp[:, 0],
        })
    return out


def analyze(label, cap, tag):
    motion = base_motion(tag)
    table = rows(f"phone_table_contact_force_*{tag}.csv")
    table_n = defaultdict(float)
    for r in table:
        table_n[round(float(r["sim_time"]), 9)] = max(
            table_n[round(float(r["sim_time"]), 9)], float(r["normal_force"])
        )
    vector = rows(f"contact_event_log_action_gripper_*{tag}.csv")
    contacts = defaultdict(list)
    seen = set()
    for r in vector:
        if "follower_left" not in r["sensor_prim"] or "/phone" not in r["other_prim"].lower():
            continue
        t = round(float(r["sim_time"]), 9)
        key = (t, r["sensor_prim"])
        if key in seen:
            continue
        seen.add(key)
        n = np.array([float(r[k]) for k in ("normal_x", "normal_y", "normal_z")])
        fr = np.array([float(r[k]) for k in ("friction_force_x", "friction_force_y", "friction_force_z")])
        force_phone = -(n * float(r["normal_force"]) + fr)
        point = np.array([float(r[k]) for k in ("contact_x", "contact_y", "contact_z")])
        side = "left" if "carriage_left" in r["sensor_prim"] else "right"
        contacts[t].append((side, point, force_phone))
    for m in motion:
        t = round(m["time"], 9)
        cs = contacts.get(t, [])
        m["table_n"] = table_n.get(t, 0.)
        m["bilateral"] = {x[0] for x in cs} == {"left", "right"}
        m["force"] = np.sum([x[2] for x in cs], axis=0) if cs else np.zeros(3)
        if cs:
            weights = np.array([max(np.linalg.norm(x[2]), 1e-12) for x in cs])
            grasp = np.average(np.array([x[1] for x in cs]), axis=0, weights=weights)
            com = m["phone"] + m["rp"] @ COM_LOCAL
            pinch = m["pinch"] / max(np.linalg.norm(m["pinch"]), 1e-12)
            tg = float(np.dot(np.cross(com-grasp, G), pinch))
            tc = float(sum(np.dot(np.cross(p-com, f), pinch) for _, p, f in cs))
            m["gravity"] = abs(tg)
            m["resist"] = abs(tc)
            m["margin"] = abs(tg)-abs(tc)
        else:
            m["gravity"] = m["resist"] = m["margin"] = math.nan
    valid_lift = lambda m: (
        m["clearance"] > .002 and m["table_n"] < .05*WEIGHT and m["bilateral"]
    )
    lift_duration = duration(motion, valid_lift)
    true_lift = lift_duration >= REQUIRED_SIM_DURATION
    lift_rows = [m for m in motion if valid_lift(m)]
    first_lift = lift_rows[0] if lift_rows else None
    post = [m for m in lift_rows if first_lift and m["time"] >= first_lift["time"]]
    relative = []
    if first_lift:
        v0 = first_lift["long_gripper"]
        relative = [math.degrees(math.acos(float(np.clip(np.dot(v0, m["long_gripper"]), -1, 1)))) for m in post]
    pivot_start = bool(true_lift and relative and max(relative) > 2 and any(m["margin"] > 0 for m in post))
    portrait_min = min((m["portrait"] for m in post), default=math.nan)
    portrait_pass = bool(true_lift and ((relative and max(relative) >= 60) or portrait_min <= 15))
    right_approach_frame = 500
    hold_pass = bool(
        portrait_pass
        and any(m["frame"] >= right_approach_frame and valid_lift(m) for m in motion)
        and not any(m["frame"] < right_approach_frame and m["frame"] > first_lift["frame"] and m["table_n"] >= .05*WEIGHT for m in motion)
    )
    grip = rows(f"gripper_action_physics_log_*{tag}.csv")
    active_grip = [r for r in grip if 280 <= int(r["source_frame"]) <= 430]
    effort = [float(r["left_drive_effort"]) for r in active_grip]
    saturated = [int(r["left_effort_saturated"]) != 0 for r in active_grip]
    exact = rows(f"left_phone_exact_shape_contacts_*{tag}.csv")
    penetration = max((
        float(r["penetration"]) for r in exact
        if "follower_left" in (r["actor0"]+r["actor1"]) and "/Phone" in (r["actor0"]+r["actor1"])
    ), default=0.)
    tracks = defaultdict(list)
    for r in exact:
        joined = r["actor0"]+r["actor1"]
        if "follower_left" in joined and "/Phone" in joined:
            actor = r["actor0"] if "follower_left" in r["actor0"] else r["actor1"]
            tracks[actor].append(np.array([float(r[k]) for k in (
                "phone_local_contact_x", "phone_local_contact_y", "phone_local_contact_z"
            )]))
    slip = max((np.max(np.linalg.norm(np.asarray(v)-v[0], axis=1)) for v in tracks.values()), default=math.nan)
    gravity = [m["gravity"] for m in post if np.isfinite(m["gravity"])]
    resist = [m["resist"] for m in post if np.isfinite(m["resist"])]
    margin = [m["margin"] for m in post if np.isfinite(m["margin"])]
    fz = [m["force"][2] for m in post]
    lock = bool(
        true_lift and percentile50(resist) >= percentile50(gravity)
        and (max(relative, default=0) < 45) and portrait_min > 30
    )
    return {
        "label": label, "tag": tag, "effort_cap_N_per_carriage": cap,
        "stiffness_N_m": 1.05*cap/.0039403,
        "applied_effort_total_P5_median_P95_N": pct(effort),
        "saturation_ratio": sum(saturated)/len(saturated),
        "saturation_duration_sim_s": sum(saturated)/60,
        "true_clearance_max_m": max(m["clearance"] for m in motion),
        "true_lift_duration_sim_s": lift_duration,
        "true_lift_duration_source_equivalent_s": lift_duration*SOURCE_SPEED,
        "table_force_released": bool(any(m["table_n"] < .05*WEIGHT for m in motion)),
        "actual_world_Z_force_P5_median_P95_N": pct(fz),
        "gravity_torque_P5_median_P95_Nm": pct(gravity),
        "resisting_torque_P5_median_P95_Nm": pct(resist),
        "net_pivot_margin_P5_median_P95_Nm": pct(margin),
        "relative_rotation_after_true_lift_deg": max(relative, default=math.nan),
        "minimum_portrait_error_after_true_lift_deg": portrait_min,
        "bilateral_lift_duration_sim_s": duration(motion, lambda m: m["clearance"]>.002 and m["bilateral"]),
        "phone_local_slip_m": float(slip), "maximum_left_phone_penetration_m": penetration,
        "phone_drop_before_right_approach": bool(first_lift and any(
            m["frame"] < right_approach_frame and m["frame"] > first_lift["frame"] and m["clearance"] <= .002
            for m in motion
        )),
        "numerically_stable": bool(penetration < .001),
        "TRUE_LIFT_PASS": true_lift, "GRAVITY_PIVOT_START_PASS": pivot_start,
        "PORTRAIT_ROTATION_PASS": portrait_pass, "SUSTAINED_PIVOT_HOLD_PASS": hold_pass,
        "ROTATION_LOCK": lock,
    }


def percentile50(v):
    return float(np.median(v)) if v else math.nan


search_stats = [analyze(*x) for x in SEARCH]
window_stats = [analyze(*x) for x in WINDOW]
all_stats = search_stats + window_stats

csv_path = OUT / "phone_effort_cap_search_runs.csv"
with csv_path.open("x", newline="", encoding="utf-8") as f:
    fields = list(all_stats[0])
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for s in all_stats:
        w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in s.items()})

for s in all_stats:
    (REPORT_DIR / f"{s['label']}_{s['effort_cap_N_per_carriage']:.6f}.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in s.items())+"\n", encoding="utf-8"
    )

lift_fail, lift_pass = 4.507623, 4.866119
lift_passes = sorted([s for s in window_stats if s["TRUE_LIFT_PASS"]], key=lambda s: s["effort_cap_N_per_carriage"])
locks = sorted([s for s in window_stats if s["ROTATION_LOCK"]], key=lambda s: s["effort_cap_N_per_carriage"])
lock_min = locks[0]["effort_cap_N_per_carriage"] if locks else 38.4175897
successes = [s for s in lift_passes if all(s[k] for k in (
    "GRAVITY_PIVOT_START_PASS", "PORTRAIT_ROTATION_PASS",
    "SUSTAINED_PIVOT_HOLD_PASS", "numerically_stable"
))]
selected = successes[0] if successes else None

(OUT / "phone_true_lift_threshold_report.txt").open("x", encoding="utf-8").write(
    f"E_lift_min_N_per_carriage={lift_pass}\n"
    f"bracket_fail_pass_N_per_carriage=[{lift_fail},{lift_pass}]\n"
    f"relative_bracket_width={(lift_pass/lift_fail-1):.9g}\n"
)
(OUT / "phone_rotation_lock_threshold_report.txt").open("x", encoding="utf-8").write(
    f"E_lock_min_N_per_carriage={lock_min}\n"
    f"bracket_N_per_carriage=[{max(s['effort_cap_N_per_carriage'] for s in window_stats)},{lock_min}]\n"
    f"hardware_max_lock_reference=True\n"
)
(OUT / "phone_lift_pivot_window_comparison.txt").open("x", encoding="utf-8").write(
    "\n".join(json.dumps(s, sort_keys=True) for s in window_stats)+"\n"
)
config = {
    "status": "PIVOT_FORCE_WINDOW_FOUND" if selected else "NO_WINDOW_IN_RIGID_CONTACT_MODEL",
    "selected_effort_cap_N_per_carriage": selected["effort_cap_N_per_carriage"] if selected else None,
    "selected_stiffness_N_m": selected["stiffness_N_m"] if selected else None,
    "damping": 10.0, "velocity_limit_m_s": .5,
    "gripper_command_mode": "hardware-max-close", "source_speed_validated": .25,
}
(OUT / "phone_selected_effort_window_config.json").open("x", encoding="utf-8").write(json.dumps(config, indent=2)+"\n")
(OUT / "phone_effort_window_final_summary.txt").open("x", encoding="utf-8").write(
    f"prior_classification=NO_PIVOT_WINDOW_FOUND_IN_TESTED_CAPS\n"
    f"PIVOT_FORCE_WINDOW_FOUND={selected is not None}\n"
    f"selected={selected['label'] if selected else 'NONE'}\n"
    f"E_lift_min={lift_pass}\nE_lock_min={lock_min}\n"
)
print(json.dumps({"search": search_stats, "window": window_stats, "selected": selected}, indent=2))
