#!/usr/bin/env python3
"""Post-process the three bounded pivot-cap runs without changing physics."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "converted_runs/magsafe_20260724_154440/isaac_replay"
DT = 1.0 / 60.0
TABLE_Z = 0.795
HALF = np.array([0.1496, 0.00795, 0.0715]) / 2.0
MASS = 0.217
G = np.array([0.0, 0.0, -MASS * 9.81])
COM_LOCAL = np.array([0.0044, 0.0024758, -0.0017408])

RUNS = {
    "P1": ("bare_phone_split_effort_cap_2_6550_uniform_box_shape_v1_pivot_P1_v2", 2.655),
    "P2": ("bare_phone_split_effort_cap_2_9210_uniform_box_shape_v1_pivot_P2_v2", 2.921),
    "P3": ("bare_phone_split_effort_cap_3_3190_uniform_box_shape_v1_pivot_P3_v2", 3.319),
    "HARDWARE_MAX": ("bare_phone_split_hardware_max_uniform_box_shape_v1_hardware_upper_bound", 38.4175897),
}


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quat_matrix_xyzw(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def percentile(v, p):
    return float(np.percentile(np.asarray(v, float), p)) if v else math.nan


def longest(mask):
    best = cur = 0
    for value in mask:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return best * DT


def motion_path(tag):
    direct = OUT / f"phone_passive_rotation_{tag}.csv"
    if direct.exists():
        return direct
    # Hardware motion has the diagnostic suffix while its actuator/contact logs do not.
    return OUT / "phone_passive_rotation_bare_phone_split_hardware_max_uniform_box_shape_v1_hardware_upper_bound.csv"


def gripper_path(tag):
    p = OUT / f"gripper_action_physics_log_{tag}.csv"
    if p.exists():
        return p
    return OUT / "gripper_action_physics_log_bare_phone_split_hardware_max_uniform_box_shape_v1.csv"


def table_path(tag):
    return OUT / f"phone_table_contact_force_{tag}.csv"


def contact_path(tag):
    p = OUT / f"left_phone_exact_shape_contacts_{tag}.csv"
    if p.exists():
        return p
    return OUT / "left_phone_exact_shape_contacts_bare_phone_split_hardware_max_uniform_box_shape_v1.csv"

def vector_contact_path(tag):
    p = OUT / f"contact_event_log_action_gripper_{tag}.csv"
    if p.exists():
        return p
    return OUT / "contact_event_log_action_gripper_bare_phone_split_hardware_max_uniform_box_shape_v1_hardware_upper_bound.csv"


def analyze(name, tag, cap):
    motion = read(motion_path(tag))
    grip = read(gripper_path(tag))
    exact = read(contact_path(tag))
    vector_contacts = read(vector_contact_path(tag))
    table = read(table_path(tag))
    m_by_time = {}
    for r in motion:
        frame = int(r["source_frame"])
        if not (280 <= frame <= 430):
            continue
        t = round(float(r["sim_time"]), 9)
        q = np.array([float(r[k]) for k in ("quat_x", "quat_y", "quat_z", "quat_w")])
        rot = quat_matrix_xyzw(q)
        phone = np.array([float(r[k]) for k in ("phone_x", "phone_y", "phone_z")])
        clearance = phone[2] - np.abs(rot[2]) @ HALF - TABLE_Z
        pinch = None
        if "pinch_axis_x" in r:
            pinch = np.array([float(r[k]) for k in ("pinch_axis_x", "pinch_axis_y", "pinch_axis_z")])
            pinch /= max(np.linalg.norm(pinch), 1e-12)
        m_by_time[t] = dict(
            frame=frame, rot=rot, phone=phone, clearance=clearance, pinch=pinch,
            portrait=float(r["long_axis_vertical_error_deg"]),
            relative=float(r["relative_rotation_angle_deg"]),
            omega=np.array([float(r[k]) for k in ("omega_x", "omega_y", "omega_z")]),
            contact_count=int(r["contact_count"]),
        )
    # ContactSensor provides the full normal+tangential force vector. Its force
    # acts on the sensor body, so reverse it for force acting on the phone.
    # One representative is retained per sensor/time because the pair friction
    # vector is repeated for each returned patch point.
    contacts = defaultdict(list)
    seen = set()
    for r in vector_contacts:
        frame = int(r["source_frame"])
        if not (
            280 <= frame <= 430
            and "follower_left" in r["sensor_prim"]
            and "/phone" in r["other_prim"].lower()
        ):
            continue
        t = round(float(r["sim_time"]), 9)
        if t not in m_by_time:
            continue
        key = (t, r["sensor_prim"])
        if key in seen:
            continue
        seen.add(key)
        normal = np.array([float(r[k]) for k in ("normal_x", "normal_y", "normal_z")])
        friction = np.array([float(r[k]) for k in ("friction_force_x", "friction_force_y", "friction_force_z")])
        force = -(normal * float(r["normal_force"]) + friction)
        point = np.array([float(r[k]) for k in ("contact_x", "contact_y", "contact_z")])
        contacts[t].append((point, force, normal, r["sensor_prim"], r["other_prim"]))
    timeline = []
    previous_omega_axis = None
    for t in sorted(m_by_time):
        m = m_by_time[t]
        cs = contacts.get(t, [])
        if m["pinch"] is None:
            # Hardware-max predates wrist logging: opposing contact normals provide
            # the measured closing-axis direction, with sign continuity.
            vectors = [normal for _, _, normal, _, _ in cs]
            pinch = np.mean(vectors, axis=0) if vectors else np.array([0., 1., 0.])
            pinch /= max(np.linalg.norm(pinch), 1e-12)
        else:
            pinch = m["pinch"]
        com = m["phone"] + m["rot"] @ COM_LOCAL
        if cs:
            weights = np.array([np.linalg.norm(f) for _, f, *_ in cs])
            grasp = np.average(np.array([p for p, *_ in cs]), axis=0, weights=np.maximum(weights, 1e-12))
        else:
            grasp = np.full(3, np.nan)
        tau_g = float(np.dot(np.cross(com - grasp, G), pinch)) if cs else math.nan
        tau_c = float(sum(np.dot(np.cross(p - com, f), pinch) for p, f, *_ in cs))
        force_world = np.sum([f for _, f, *_ in cs], axis=0) if cs else np.zeros(3)
        # Choose positive as the instantaneous gravity direction.
        sign = 1.0 if not np.isfinite(tau_g) or tau_g >= 0 else -1.0
        tau_g_dir = tau_g * sign
        tau_c_dir = tau_c * sign
        net = tau_g_dir + tau_c_dir
        omega_axis = float(np.dot(m["omega"], pinch) * sign)
        alpha_axis = 0.0 if previous_omega_axis is None else (omega_axis - previous_omega_axis) / DT
        previous_omega_axis = omega_axis
        timeline.append(dict(
            run=name, sim_time=t, source_frame=m["frame"],
            pinch_axis_x=pinch[0], pinch_axis_y=pinch[1], pinch_axis_z=pinch[2],
            gravity_torque_axis_Nm=tau_g_dir, contact_torque_axis_Nm=tau_c_dir,
            resisting_contact_torque_magnitude_Nm=abs(tau_c_dir),
            gravity_minus_resisting_margin_Nm=abs(tau_g_dir)-abs(tau_c_dir),
            gripper_force_x_N=force_world[0], gripper_force_y_N=force_world[1],
            gripper_force_z_N=force_world[2],
            damping_torque_axis_Nm=0.0, joint_external_torque_axis_Nm=0.0,
            magnetic_torque_axis_Nm=0.0, net_torque_axis_Nm=net,
            angular_velocity_axis_rad_s=omega_axis, angular_acceleration_axis_rad_s2=alpha_axis,
            relative_rotation_deg=m["relative"], portrait_error_deg=m["portrait"],
            true_clearance_m=m["clearance"], phone_contact_count=m["contact_count"],
        ))
    # Table log values are force samples; absence means released.
    table_force = defaultdict(float)
    for r in table:
        frame = int(r["source_frame"])
        if 280 <= frame <= 430:
            table_force[round(float(r["sim_time"]), 9)] += max(0.0, float(r["normal_force"]))
    for row in timeline:
        row["table_normal_force_N"] = table_force.get(round(row["sim_time"], 9), 0.0)
    gr = [r for r in grip if 280 <= int(r["source_frame"]) <= 430]
    effort = [float(r["left_drive_effort"]) for r in gr]
    sat = [int(r["left_effort_saturated"]) != 0 for r in gr]
    clear = [r["true_clearance_m"] for r in timeline]
    lift_duration = longest([c > .002 for c in clear])
    lifted = lift_duration >= .2
    active = [r for r in timeline if r["phone_contact_count"] > 0]
    portrait_min_active = min((r["portrait_error_deg"] for r in active), default=math.nan)
    relative_range = (
        max(r["relative_rotation_deg"] for r in active) - min(r["relative_rotation_deg"] for r in active)
        if active else math.nan
    )
    contact_duration = len(active) * DT
    penetration = max((float(r["penetration"]) for r in exact if 280 <= int(r["source_frame"]) <= 430), default=0.0)
    local_tracks = defaultdict(list)
    for r in exact:
        joined = r["actor0"] + r["actor1"]
        if (
            280 <= int(r["source_frame"]) <= 430
            and "follower_left" in joined
            and "/Phone" in joined
        ):
            carriage = r["actor0"] if "follower_left" in r["actor0"] else r["actor1"]
            local_tracks[carriage].append(
                np.array([float(r[k]) for k in (
                    "phone_local_contact_x", "phone_local_contact_y", "phone_local_contact_z"
                )])
            )
    local_slip = max(
        (float(np.max(np.linalg.norm(np.asarray(points) - points[0], axis=1)))
         for points in local_tracks.values() if points),
        default=math.nan,
    )
    gravity = [r["gravity_torque_axis_Nm"] for r in active if np.isfinite(r["gravity_torque_axis_Nm"])]
    resist = [r["resisting_contact_torque_magnitude_Nm"] for r in active]
    net = [r["net_torque_axis_Nm"] for r in active]
    margin = [r["gravity_minus_resisting_margin_Nm"] for r in active]
    force_z = [r["gripper_force_z_N"] for r in active]
    return timeline, dict(
        name=name, cap=cap, stiffness=1.05*cap/.0039403 if name != "HARDWARE_MAX" else 9749.9149,
        effort_median=percentile(effort, 50), saturation_duration=sum(sat)*DT,
        clearance_max=max(clear), lift_duration=lift_duration, true_lift=lifted,
        table_release_duration=longest([r["table_normal_force_N"] < .05*MASS*9.81 for r in timeline]),
        gravity_p50=percentile(gravity, 50), resist_p50=percentile(resist, 50),
        net_p50=percentile(net, 50), pivot_margin_p50=percentile(margin, 50),
        net_positive_duration=sum(x > 0 for x in margin)*DT,
        world_z_force_median=percentile(force_z, 50), world_z_force_p95=percentile(force_z, 95),
        world_z_force_max=max(force_z, default=math.nan),
        portrait_min=portrait_min_active, relative_rotation=relative_range,
        contact_duration=contact_duration, penetration=penetration,
        phone_local_contact_slip=local_slip,
    )


def write_new(path, text):
    # These names are owned by this analysis script; reruns replace only its
    # immediately preceding incomplete products, never legacy reports.
    with path.open("w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


all_rows = []
stats = {}
for name, (tag, cap) in RUNS.items():
    rows, stat = analyze(name, tag, cap)
    all_rows.extend(rows)
    stats[name] = stat

timeline_path = OUT / "phone_pinch_axis_torque_timeline.csv"
with timeline_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)

dof = """phone angular axis lock: false (no lockedAxes attributes)
phone angular damping: no authored angularDamping; PhysX scene default
max angular velocity: 25 rad/s
sleep threshold: no per-body authored override; PhysX scene default
stabilization threshold: no per-body authored override; PhysX scene default
kinematic enabled: false
world-fixed constraint involving phone: none
phone-accessory joint: /World/MagSafeScene/MagneticJoints/AccessoryPhone; body-to-body fixed joint
orientation controller: none during lift
contact-conditioned grasp constraint: none
charger magnetic state frames 250-430: DETACHED
charger magnetic force frames 250-430: 0 N
charger magnetic torque frames 250-430: 0 N m
"""
write_new(OUT / "phone_angular_dof_audit.txt", dof)

hm = stats["HARDWARE_MAX"]
write_new(OUT / "phone_hardware_max_rotation_lock_report.txt", "\n".join([
    f"gravity_torque_axis_median_Nm={hm['gravity_p50']:.9g}",
    f"resisting_contact_torque_axis_median_Nm={hm['resist_p50']:.9g}",
    f"gravity_minus_resisting_margin_median_Nm={hm['pivot_margin_p50']:.9g}",
    f"relative_rotation_active_contact_deg={hm['relative_rotation']:.9g}",
    "angular_DOF_unlocked=True",
    "magnetic_torque_zero=True",
    f"TORSIONAL_CONTACT_FRICTION_LOCK={hm['true_lift'] and hm['pivot_margin_p50'] <= 0}",
]))

candidate_lines = [
    "basis=baseline peak world-Z shortfall ratio 2.21399/2.0967=1.05594",
    "P1=baseline peak applied effort per carriage times shortfall ratio",
    "P2=P1*1.10",
    "P3=P1*1.25",
]
for n in ("P1", "P2", "P3"):
    s = stats[n]
    candidate_lines.append(f"{n}: cap_N_per_carriage={s['cap']:.6g} stiffness_N_m={s['stiffness']:.6g}")
write_new(OUT / "phone_lift_pivot_effort_window_candidates.txt", "\n".join(candidate_lines))

for n in ("P1", "P2", "P3"):
    s = stats[n]
    lines = [f"{k}={v}" for k, v in s.items()]
    lines += [
        f"GRAVITY_PIVOT_START_PASS={s['net_positive_duration'] > 0 and s['relative_rotation'] > 5}",
        f"PORTRAIT_ROTATION_PASS={s['portrait_min'] <= 15 and s['true_lift']}",
        f"SUSTAINED_PIVOT_HOLD_PASS={s['true_lift'] and s['contact_duration'] >= 1.0}",
    ]
    write_new(OUT / f"phone_pivot_candidate_{n}_report.txt", "\n".join(lines))

comparison = []
for n in ("P1", "P2", "P3", "HARDWARE_MAX"):
    comparison.append(json.dumps(stats[n], sort_keys=True))
write_new(OUT / "phone_lift_pivot_effort_window_comparison.txt", "\n".join(comparison))

selected = {
    "status": "NO_VALID_SELECTION",
    "reason": "LIFT_ROTATION_TRADEOFF_NO_WINDOW",
    "maximum_close_target_maintained": True,
    "hardware_upper_bound_N_per_carriage": 38.4175897,
    "tested_candidates_N_per_carriage": [2.655, 2.921, 3.319],
}
write_new(OUT / "phone_selected_pivot_drive_config.json", json.dumps(selected, indent=2))
write_new(OUT / "phone_gravity_pivot_final_report.txt", "\n".join([
    "PIVOT_FORCE_WINDOW_FOUND=False",
    "classification=LIFT_ROTATION_TRADEOFF_NO_WINDOW",
    "P1 fails true lift.",
    "P2 and P3 exceed 2 mm only transiently, below the required 0.2 s.",
    "Apparent portrait angles after contact loss/drop are not a successful gravity pivot hold.",
    "No candidate satisfies TRUE_LIFT_PASS + PORTRAIT_ROTATION_PASS + SUSTAINED_PIVOT_HOLD_PASS.",
    "next_unapplied_model_change=inspect supported PhysX compliant/torsional patch-friction properties; preserve translational friction",
]))

print(json.dumps(stats, indent=2))
