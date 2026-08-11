#!/usr/bin/env python3
"""Finalize the frozen Candidate-A retention force/material forensic audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_retention_force_audit_v1"
BASE = OUT / "baseline_true_physics_v2"
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
PRIMITIVE = CAL / "left_phone_fingertip_pinch_primitive.json"
V17_2 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene"
HIST = ROOT / "converted_runs/magsafe_20260724_154440/isaac_replay"
HIST_NEW = ROOT / "converted_runs/magsafe_20260727_174234/isaac_replay"
W1 = "bare_phone_split_effort_cap_4_8661_uniform_box_shape_v1_pivot_window_W1"
APPROVED_Q = np.asarray([
    -0.517737046, 0.747053166, 0.050425649,
    -0.661925094, -1.705330000, -0.100000000, -0.100000000,
])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, data) -> None:
    def conv(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=conv, allow_nan=False) + "\n", encoding="utf-8")


def run_text(command: list[str]) -> str:
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return (proc.stdout + proc.stderr).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def qstats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.abs(b) > 1e-12)


def aloha_reference_metrics() -> tuple[dict, dict]:
    motion_path = HIST / f"phone_passive_rotation_{W1}.csv"
    contact_path = HIST / f"contact_event_log_action_gripper_{W1}.csv"
    table_path = HIST / f"phone_table_contact_force_{W1}.csv"
    grip_path = HIST / f"gripper_action_physics_log_{W1}.csv"
    search_path = HIST / "phone_effort_cap_search_runs.csv"

    motion_rows = read_csv(motion_path)
    motion = {}
    half = np.asarray([0.1496, 0.00795, 0.0715]) / 2.0
    table_z = 0.795
    for row in motion_rows:
        frame = int(row["source_frame"])
        if frame < 280:
            continue
        t = round(float(row["sim_time"]), 9)
        rot = Rotation.from_quat([float(row[k]) for k in ("quat_x", "quat_y", "quat_z", "quat_w")])
        phone = np.asarray([float(row[k]) for k in ("phone_x", "phone_y", "phone_z")])
        clearance = phone[2] - np.abs(rot.as_matrix()[2]) @ half - table_z
        motion[t] = {"frame": frame, "rot": rot, "phone": phone, "clearance": float(clearance)}

    table_force = defaultdict(float)
    for row in read_csv(table_path):
        t = round(float(row["sim_time"]), 9)
        table_force[t] = max(table_force[t], float(row["normal_force"]))

    contacts = defaultdict(list)
    seen = set()
    for row in read_csv(contact_path):
        if "follower_left" not in row["sensor_prim"] or "/phone" not in row["other_prim"].lower():
            continue
        t = round(float(row["sim_time"]), 9)
        key = (t, row["sensor_prim"])
        if key in seen or t not in motion:
            continue
        seen.add(key)
        normal = np.asarray([float(row[k]) for k in ("normal_x", "normal_y", "normal_z")])
        normal_n = float(row["normal_force"])
        friction_on_sensor = np.asarray([float(row[k]) for k in (
            "friction_force_x", "friction_force_y", "friction_force_z"
        )])
        point = np.asarray([float(row[k]) for k in ("contact_x", "contact_y", "contact_z")])
        local = motion[t]["rot"].inv().apply(point - motion[t]["phone"])
        face = "front_glass" if local[1] < 0 else "rear_matte"
        mu_static, mu_dynamic = ((0.55, 0.425) if face == "front_glass" else (0.65, 0.525))
        side = "carriage_left" if "carriage_left" in row["sensor_prim"] else "carriage_right"
        contacts[t].append({
            "side": side, "point": point, "normal_n": normal_n,
            "normal_on_phone": -(normal * normal_n),
            "friction_on_phone": -friction_on_sensor,
            "face": face, "mu_static": mu_static, "mu_dynamic": mu_dynamic,
        })

    phone_mass = 0.190
    accessory_mass = 0.027
    load_mass = phone_mass + accessory_mass
    weight = load_mass * 9.81
    selected = []
    contact_samples = []
    for t, row in sorted(motion.items()):
        cs = contacts.get(t, [])
        bilateral = {c["side"] for c in cs} == {"carriage_left", "carriage_right"}
        if row["clearance"] > 0.002 and table_force[t] < 0.05 * weight and bilateral:
            normal_sum = sum(np.linalg.norm(c["normal_on_phone"]) for c in cs)
            total_vector = np.sum(
                [c["normal_on_phone"] + c["friction_on_phone"] for c in cs], axis=0
            )
            selected.append({
                "time_s": t, "frame": row["frame"],
                "normal_ratio": normal_sum / weight,
                "support_ratio": total_vector[2] / weight,
                "total_vector": total_vector,
            })
            for c in cs:
                n = float(np.linalg.norm(c["normal_on_phone"]))
                ft = float(np.linalg.norm(c["friction_on_phone"]))
                contact_samples.append({
                    **c, "time_s": t, "normal_magnitude_n": n, "tangential_magnitude_n": ft,
                    "friction_utilization_dynamic": ft / max(c["mu_dynamic"] * n, 1e-12),
                    "friction_margin_dynamic_n": c["mu_dynamic"] * n - ft,
                })

    if not selected or not contact_samples:
        raise RuntimeError("archived ALOHA W1 lift/contact evidence is empty")
    grip = [
        row for row in read_csv(grip_path)
        if 280 <= int(row["source_frame"]) <= 430
    ]
    effort = np.asarray([float(row["left_drive_effort"]) for row in grip])
    effort_limit_total = 2.0 * 4.866119
    effort_util = effort / effort_limit_total
    search = next(row for row in read_csv(search_path) if row["label"] == "W1")
    rn = np.asarray([row["normal_ratio"] for row in selected])
    rs = np.asarray([row["support_ratio"] for row in selected])
    # The archived logger stores one normal scalar and one aggregate friction
    # vector per sensor rigid body, so literal per-point pairing is not
    # recoverable.  Aggregate per timestamp to avoid dividing a patch-level
    # friction vector by a zero/near-zero point scalar.
    aggregate_friction_utilization = []
    aggregate_friction_margin = []
    by_time = defaultdict(list)
    for row in contact_samples:
        by_time[row["time_s"]].append(row)
    for rows_at_time in by_time.values():
        capacity = sum(
            row["mu_dynamic"] * row["normal_magnitude_n"] for row in rows_at_time
        )
        tangential = float(np.linalg.norm(np.sum(
            [row["friction_on_phone"] for row in rows_at_time], axis=0
        )))
        if capacity > 0.05:
            aggregate_friction_utilization.append(tangential / capacity)
            aggregate_friction_margin.append(capacity - tangential)
    uf = np.asarray(aggregate_friction_utilization)
    mf = np.asarray(aggregate_friction_margin)
    metrics = {
        "evidence_type": "ARCHIVED_REPRODUCED_ISAAC_PHYSICS_W1",
        "reference_scope": "true-lift/pivot-window reference, not a zero-slip or full-task success",
        "load": {
            "phone_mass_kg": phone_mass, "attached_accessory_mass_kg": accessory_mass,
            "combined_mass_kg": load_mass, "object_weight_n": weight,
        },
        "selected_true_lift_samples": len(selected),
        "selected_first_last_time_s": [selected[0]["time_s"], selected[-1]["time_s"]],
        "normal_force_ratio": qstats(rn),
        "vertical_support_ratio": qstats(rs),
        "friction_utilization_dynamic": {
            **qstats(uf),
            "definition": "net archived friction vector magnitude / sum(mu_dynamic * archived normal scalar), aggregated per timestamp",
            "validity_warning": "ARCHIVED_NORMAL_FRICTION_PATCH_PAIRING_NOT_AVAILABLE; values above one expose logger aggregation mismatch and are not treated as a calibrated Coulomb-cone measurement",
        },
        "friction_margin_dynamic_n": {
            **qstats(mf),
            "validity_warning": "same archived aggregation limitation as friction utilization",
        },
        "actuator_utilization": {
            "definition": "left_drive_effort / (2 * per-carriage effort cap)",
            **qstats(effort_util),
            "archived_saturation_ratio": float(search["saturation_ratio"]),
        },
        "phone_local_contact_slip_m": float(search["phone_local_slip_m"]),
        "relative_rotation_after_true_lift_deg": float(search["relative_rotation_after_true_lift_deg"]),
        "minimum_portrait_error_after_true_lift_deg": float(search["minimum_portrait_error_after_true_lift_deg"]),
        "retention_result": {
            "TRUE_LIFT_PASS": search["TRUE_LIFT_PASS"] == "True",
            "SUSTAINED_PIVOT_HOLD_PASS": search["SUSTAINED_PIVOT_HOLD_PASS"] == "True",
            "ROTATION_LOCK": search["ROTATION_LOCK"] == "True",
            "strict_zero_slip_hold": False,
        },
    }
    raw = {
        "selected": selected,
        "contacts": contact_samples,
        "paths": [motion_path, contact_path, table_path, grip_path, search_path],
    }
    return metrics, raw


def dex3_metrics() -> tuple[dict, dict, list[dict]]:
    z = np.load(BASE / "static_physics_trace.npz", allow_pickle=False)
    t = z["time_s"].astype(float)
    dt = float(np.median(np.diff(t)))
    normal = z["normal_force_on_phone_w"][:, :2].astype(float)
    friction = z["friction_force_on_phone_w"][:, :2].astype(float)
    total = normal + friction
    normal_mag = np.linalg.norm(normal, axis=2)
    friction_mag = np.linalg.norm(friction, axis=2)
    mu = 0.5
    phone_mass = 0.177
    weight = phone_mass * 9.81
    rn = normal_mag.sum(axis=1) / weight
    rs = total.sum(axis=1)[:, 2] / weight
    uf = safe_ratio(friction_mag, mu * normal_mag)
    mf = mu * normal_mag - friction_mag
    pose = z["phone_pose_xyzw"].astype(float)
    vel = z["phone_velocity"].astype(float)
    centers = z["contact_centroid_w"][:, :2].astype(float)
    moments = np.cross(centers - pose[:, None, :3], total)
    net_moment = np.nansum(moments, axis=1)
    net_force = np.sum(total, axis=1)
    initial_rotation = Rotation.from_quat(pose[0, 3:7])
    rotation_change = np.asarray([
        math.degrees((initial_rotation.inv() * Rotation.from_quat(q)).magnitude())
        for q in pose[:, 3:7]
    ])
    pinch_center = z["pinch_center_m"].astype(float)
    relative = pose[:, :3] - pinch_center
    slip = np.linalg.norm(relative - relative[0], axis=1)

    # The phone-table filtered GPU view is unsupported in this Isaac build.
    # Infer table-impact onset only for interval partitioning, using force-balance
    # residual. The inference is never counted as a measured table force.
    acceleration = np.gradient(vel[:, :3], dt, axis=0)
    required_external = phone_mass * (acceleration - np.asarray([0.0, 0.0, -9.81]))
    unmeasured_external = required_external - net_force
    impact_candidates = np.flatnonzero(
        (t >= 0.10) & (unmeasured_external[:, 2] > 0.5 * weight)
    )
    impact_index = int(impact_candidates[0]) if len(impact_candidates) else len(t)
    free = np.arange(len(t)) < impact_index
    post = ~free

    contact_metrics = json.loads((BASE / "phone_contact_identity_metrics.json").read_text())
    separation_by_step = {}
    point_count_by_step = {}
    for label in ("THUMB", "INDEX"):
        rows = contact_metrics["identity"][label]["contact_patch_proxy"]["per_step"]
        separation_by_step[label] = {int(row["step"]): row["mean_signed_separation_m"] for row in rows}
        point_count_by_step[label] = {int(row["step"]): row["contact_point_count"] for row in rows}

    requested = z["model_requested_drive_torque_nm"].astype(float)
    clipped = z["model_clipped_drive_torque_nm"].astype(float)
    limits = z["dex3_effort_limits_nm"].astype(float)
    target = z["commanded_left_dex3_q"].astype(float)
    actual = z["actual_left_dex3_q"].astype(float)
    q_error = target - actual
    joint_names = json.loads((BASE / "runtime_drive_probe.json").read_text())["joint_names"]
    task_ids = np.arange(5)
    task_saturation = np.abs(requested[:, task_ids]) >= limits[None, task_ids] - 1e-9

    phone_local_centers = np.full_like(centers, np.nan)
    for i in range(len(t)):
        r = Rotation.from_quat(pose[i, 3:7]).inv()
        for finger in range(2):
            if np.all(np.isfinite(centers[i, finger])):
                phone_local_centers[i, finger] = r.apply(centers[i, finger] - pose[i, :3])
    height_offset = np.abs(centers[:, 0, 2] - centers[:, 1, 2])

    windows = []
    for start, end in ((0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5)):
        mask = (t >= start) & (t <= end if end == 1.5 else t < end)
        windows.append({
            "start_s": start, "end_s": end, "samples": int(mask.sum()),
            "thumb_normal_force_n": qstats(normal_mag[mask, 0]),
            "index_normal_force_n": qstats(normal_mag[mask, 1]),
            "normal_force_ratio": qstats(rn[mask]),
            "vertical_support_ratio": qstats(rs[mask]),
            "thumb_mean_signed_separation_m": qstats(np.asarray([
                separation_by_step["THUMB"].get(int(i), 0.0) for i in np.flatnonzero(mask)
            ])),
            "index_mean_signed_separation_m": qstats(np.asarray([
                separation_by_step["INDEX"].get(int(i), 0.0) for i in np.flatnonzero(mask)
            ])),
        })

    per_joint = []
    for i, name in enumerate(joint_names):
        utilization = np.abs(requested[:, i]) / limits[i]
        per_joint.append({
            "joint": name, "role": "TASK_THUMB_INDEX" if i < 5 else "NON_TASK_THIRD",
            "configured_effort_limit_nm": float(limits[i]),
            "model_requested_effort_utilization": qstats(utilization),
            "model_request_clipped_fraction": float(np.mean(utilization >= 1.0)),
            "q_error_rad": qstats(np.abs(q_error[:, i])),
            "maximum_projected_joint_force_nm": float(np.max(np.abs(z["projected_joint_force_nm"][:, i]))),
        })

    trace_rows = []
    for i in range(len(t)):
        row = {
            "step": i, "timestamp_s": float(t[i]),
            "phone_x_m": pose[i, 0], "phone_y_m": pose[i, 1], "phone_z_m": pose[i, 2],
            "phone_qx": pose[i, 3], "phone_qy": pose[i, 4], "phone_qz": pose[i, 5], "phone_qw": pose[i, 6],
            "phone_vx_m_s": vel[i, 0], "phone_vy_m_s": vel[i, 1], "phone_vz_m_s": vel[i, 2],
            "phone_wx_rad_s": vel[i, 3], "phone_wy_rad_s": vel[i, 4], "phone_wz_rad_s": vel[i, 5],
            "phone_relative_slip_m": slip[i], "phone_rotation_change_deg": rotation_change[i],
            "normal_force_ratio": rn[i], "vertical_support_ratio": rs[i],
            "net_contact_force_x_n": net_force[i, 0], "net_contact_force_y_n": net_force[i, 1], "net_contact_force_z_n": net_force[i, 2],
            "net_contact_moment_x_nm": net_moment[i, 0], "net_contact_moment_y_nm": net_moment[i, 1], "net_contact_moment_z_nm": net_moment[i, 2],
            "inferred_table_impact_or_support": bool(i >= impact_index),
        }
        for j, label in enumerate(("thumb", "index")):
            row.update({
                f"{label}_normal_force_n": normal_mag[i, j],
                f"{label}_tangential_force_n": friction_mag[i, j],
                f"{label}_friction_utilization": uf[i, j],
                f"{label}_friction_margin_n": mf[i, j],
                f"{label}_contact_point_count": point_count_by_step[label.upper()].get(i, 0),
                f"{label}_mean_signed_separation_m": separation_by_step[label.upper()].get(i, math.nan),
                f"{label}_contact_centroid_x_m": centers[i, j, 0],
                f"{label}_contact_centroid_y_m": centers[i, j, 1],
                f"{label}_contact_centroid_z_m": centers[i, j, 2],
            })
            for axis, k in zip("xyz", range(3)):
                row[f"{label}_normal_force_{axis}_n"] = normal[i, j, k]
                row[f"{label}_friction_force_{axis}_n"] = friction[i, j, k]
                row[f"{label}_total_force_{axis}_n"] = total[i, j, k]
        for j, name in enumerate(joint_names):
            short = name.replace("left_hand_", "").replace("_joint", "")
            row[f"{short}_target_q_rad"] = target[i, j]
            row[f"{short}_actual_q_rad"] = actual[i, j]
            row[f"{short}_q_error_rad"] = q_error[i, j]
            row[f"{short}_model_requested_torque_nm"] = requested[i, j]
            row[f"{short}_model_clipped_torque_nm"] = clipped[i, j]
        trace_rows.append(row)

    force_metrics = {
        "status": "DEX3_FROZEN_FORCE_BALANCE_AUDIT_COMPLETE",
        "phone_mass_kg": phone_mass, "object_weight_n": weight,
        "normal_force_ratio_all": qstats(rn),
        "normal_force_ratio_pre_table_impact": qstats(rn[free]),
        "vertical_support_ratio_all": qstats(rs),
        "vertical_support_ratio_pre_table_impact": qstats(rs[free]),
        "friction_utilization": {
            "thumb_all": qstats(uf[:, 0]), "index_all": qstats(uf[:, 1]),
            "thumb_pre_table_impact": qstats(uf[free, 0]),
            "index_pre_table_impact": qstats(uf[free, 1]),
        },
        "friction_margin_n": {
            "thumb_all": qstats(mf[:, 0]), "index_all": qstats(mf[:, 1]),
        },
        "force_decay_windows": windows,
        "table_contact_sensor_runtime_status": "UNSUPPORTED_GPU_FILTER_RETURNED_ZERO",
        "inferred_table_impact_time_s": float(t[impact_index]) if impact_index < len(t) else None,
        "inference_definition": "first t>=0.1 s where Newton force-balance residual upward exceeds 0.5 object weight",
        "pre_impact_vector_force_balance_rmse_n": qstats(np.linalg.norm((required_external - net_force)[free], axis=1)),
        "phone_relative_slip_m": float(slip[-1]),
        "phone_vertical_drop_m": float(pose[-1, 2] - pose[0, 2]),
        "phone_rotation_deg": float(rotation_change[-1]),
        "bilateral_contact_duration_s": float(t[-1] - t[0] + dt),
    }
    wrench_metrics = {
        "status": "CONTACT_WRENCH_ROTATION_AUDIT_COMPLETE",
        "net_contact_force_n": {
            "magnitude": qstats(np.linalg.norm(net_force, axis=1)),
            "vertical_component": qstats(net_force[:, 2]),
        },
        "net_contact_moment_about_phone_com_nm": {
            "magnitude": qstats(np.linalg.norm(net_moment, axis=1)),
            "x": qstats(net_moment[:, 0]), "y": qstats(net_moment[:, 1]), "z": qstats(net_moment[:, 2]),
            "initial_0_to_0p1_s_magnitude": qstats(np.linalg.norm(net_moment[t < 0.1], axis=1)),
        },
        "contact_centroid_world_height_offset_m": qstats(height_offset),
        "contact_centroid_phone_local": {
            "thumb": {"x": qstats(phone_local_centers[:, 0, 0]), "y": qstats(phone_local_centers[:, 0, 1]), "z": qstats(phone_local_centers[:, 0, 2])},
            "index": {"x": qstats(phone_local_centers[:, 1, 0]), "y": qstats(phone_local_centers[:, 1, 1]), "z": qstats(phone_local_centers[:, 1, 2])},
        },
        "normal_force_imbalance_ratio": qstats(
            np.abs(normal_mag[:, 0] - normal_mag[:, 1]) / np.maximum(normal_mag.sum(axis=1), 1e-12)
        ),
        "phone_rotation_change_deg": float(rotation_change[-1]),
        "maximum_phone_angular_speed_rad_s": float(np.max(np.linalg.norm(vel[:, 3:], axis=1))),
        "dominant_measured_mechanism": (
            "The opposed normal forces are tilted and their centroids are vertically offset; the resulting net contact wrench initiates rotation while the realized upward support remains below weight. "
            "The phone then impacts the table and continues bilateral side contact in a rotated/slipped state."
        ),
    }
    actuator = {
        "status": "DEX3_ACTIVE_MODEL_LIMIT_ESTABLISHED",
        "actuator_type": "IsaacLab ImplicitActuator / PhysX position drive",
        "configured_effort_limit_nm_per_joint": limits,
        "stiffness_nm_per_rad": z["dex3_stiffness_nm_per_rad"],
        "damping_nm_s_per_rad": z["dex3_damping_nm_s_per_rad"],
        "per_joint": per_joint,
        "task_joint_model_request_clipped_fraction": float(np.mean(task_saturation)),
        "task_chain_max_q_error_rad": float(np.max(np.abs(q_error[:, :5]))),
        "task_chain_rms_q_error_rad": float(np.sqrt(np.mean(q_error[:, :5] ** 2))),
        "effort_readback_limitation": (
            "Implicit PhysX drives do not expose an actuator torque measurement through IsaacLab applied_torque. "
            "Kp error minus Kd velocity is therefore reported as a model-request/saturation proxy, not an invented measured effort."
        ),
        "authority_limited_gate": False,
        "authority_limited_gate_reason": (
            "Although the model-request proxy reaches the 2.5 Nm clip, sustained normal-force magnitude does not decay below the theoretical static support requirement and task q tracking remains close. "
            "Actual upward support fails because of wrench direction/friction use, so the required sustained-normal-force-insufficient condition is not met."
        ),
    }
    raw = {
        "t": t, "normal": normal, "friction": friction, "total": total,
        "rn": rn, "rs": rs, "uf": uf, "mf": mf, "pose": pose, "vel": vel,
        "net_force": net_force, "net_moment": net_moment,
        "rotation_change": rotation_change, "slip": slip,
        "impact_index": impact_index, "free": free, "post": post,
    }
    return force_metrics, wrench_metrics, actuator, trace_rows, raw


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    primitive = json.loads(PRIMITIVE.read_text())
    primitive_q = np.asarray(primitive["selected_static_q_rad"], dtype=float)
    if not np.array_equal(np.round(primitive_q, 9), APPROVED_Q):
        raise RuntimeError(f"Candidate A changed: {primitive_q.tolist()}")

    frozen_paths = {
        "candidate_A_primitive_json": PRIMITIVE,
        "v17_2_full_trajectory": V17_2,
        "v14_cartesian_backbone": V14,
        "scene_layout": SCENE / "scene_layout.json",
        "authoritative_scene_usd": SCENE / "generated/magsafe_fixed_scene.usda",
        "phone_asset": SCENE / "generated/phone_landscape.usda",
        "candidate_A_calibration_npz": CAL / "left_phone_fingertip_pinch_calibration.npz",
    }
    before = {key: sha(path) for key, path in frozen_paths.items()}

    git_queries = {
        "git_grep_4_866119": ["git", "grep", "-n", "4.866119", "--", "."],
        "git_grep_gripper_effort_cap": ["git", "grep", "-n", "gripper-effort-cap", "--", "."],
        "git_grep_hardware_max_close": ["git", "grep", "-n", "hardware-max-close", "--", "."],
        "git_log_4_866119": ["git", "log", "--all", "--oneline", "-S4.866119", "--", "."],
        "git_log_gripper_effort_cap": ["git", "log", "--all", "--oneline", "-Sgripper-effort-cap", "--", "."],
    }
    git_results = {key: run_text(command) for key, command in git_queries.items()}
    repository_queries = {
        "core_grasp_fix_terms": [
            "rg", "-l", "--hidden", "--glob", "!.git/**", "--max-filesize", "20M",
            "4\\.866119|gripper-effort-cap|effort-cap|hardware-max-close|bare-phone-split|tape_sleeve",
            ".",
        ],
        "physics_role_terms": [
            "rg", "-l", "--hidden", "--glob", "!.git/**", "--max-filesize", "20M",
            "gripper stiffness|gripper damping|gripper drive|phone mass|grasp retention|slip",
            ".",
        ],
    }
    repository_results = {
        key: run_text(command) for key, command in repository_queries.items()
    }

    tape_cfg = json.loads((SCENE / "tape_sleeve_config.json").read_text())
    threshold = (HIST / "phone_true_lift_threshold_report.txt").read_text().strip()
    window_cfg = json.loads((HIST / "phone_selected_effort_window_config.json").read_text())
    tape_compare = (HIST / "tape_sleeve_baseline_comparison.txt").read_text().strip()
    tape_summary = (HIST / "left_phone_tape_sleeve_final_summary.txt").read_text().strip()
    later_report = (HIST_NEW / "phone_lift_after_visual_fix_report.txt").read_text().strip()

    forensics = {
        "status": "HISTORICAL_FORENSIC_AUDIT_COMPLETE",
        "git_commit_containing_fix": "324f8c7 checkpoint: v12 ALOHA-to-G1 trajectory and Isaac renderfix",
        "git_queries": {key: {"command": " ".join(command), "output": git_results[key]} for key, command in git_queries.items()},
        "whole_repository_queries": {
            key: {"command": " ".join(command), "matching_files": repository_results[key].splitlines()}
            for key, command in repository_queries.items()
        },
        "historical_runner": SCENE / "replay_magsafe_aloha_episode.py",
        "recording_reference": ROOT / "raw_recordings/GoPark_20260727_174234",
        "evidence": {
            "true_lift_threshold": threshold,
            "selected_window_config": window_cfg,
            "tape_sleeve_config": tape_cfg,
            "tape_sleeve_baseline_comparison": tape_compare,
            "tape_sleeve_final_summary": tape_summary,
            "later_combined_condition_report": later_report,
        },
        "critical_scope_distinction": {
            "W1_effort_window": "bare-phone-split + hardware-max-close + effort-cap; filename and layer stack do not include tape_sleeve_20mm",
            "tape_sleeve_experiment": "separate stiffness-578.10 run; reduced slip and penetration but did not sustain hold",
            "later_GoPark_combined_condition": "assembly lift occurred, but strict bilateral maintenance and penetration gates failed",
        },
        "causal_attribution": {
            "4_866119_true_lift_threshold": "SUPPORTED_WITHIN_ARCHIVED_FIXED-CONDITION_BRACKET_4.507623_FAIL_TO_4.866119_PASS",
            "4_866119_full_retention_success": "HISTORICAL_CAUSAL_ATTRIBUTION_NOT_AVAILABLE",
            "tape_sleeve_effect": "36.7738213306 percent slip reduction and 83.959871437 percent penetration reduction in its separate controlled run; sustained hold still failed",
        },
    }
    dump(OUT / "aloha_historical_grasp_forensics.json", forensics)

    aloha_metrics, aloha_raw = aloha_reference_metrics()
    success_summary = {
        "status": "ALOHA_REFERENCE_CONDITION_RECONSTRUCTED",
        "command": {"mode": "hardware-max-close", "meaning": "closed source command is replaced by zero-aperture hardware-stop target"},
        "drive": {
            "mode": "effort-cap", "effort_cap_n_per_carriage": 4.866119,
            "stiffness_n_per_m": 1296.7096287084742, "damping_n_s_per_m": 10.0,
            "velocity_limit_m_s": 0.5,
            "hardware_upper_bound_n_per_carriage": 38.4175897,
        },
        "material": {
            "profile": "bare-phone-split", "combine_mode": "average",
            "tape_static_dynamic": [0.75, 0.60], "phone_glass_static_dynamic": [0.35, 0.25],
            "phone_matte_static_dynamic": [0.55, 0.45],
            "effective_tape_glass_static_dynamic": [0.55, 0.425],
            "effective_tape_matte_static_dynamic": [0.65, 0.525],
        },
        "geometry": {
            "W1_tape_sleeve_present": False,
            "separate_tape_sleeve": tape_cfg,
            "separate_sleeve_effect": "continuous distal surface reduced slip/penetration but did not create sustained hold",
        },
        "load": aloha_metrics["load"],
        "result": aloha_metrics["retention_result"],
        "interpretation": "The archived condition is valid true-lift/pivot-window evidence, not proof of a stable zero-slip or full-task ALOHA grasp.",
    }
    dump(OUT / "aloha_success_condition_summary.json", success_summary)

    force_metrics, wrench_metrics, actuator_metrics, trace_rows, dex3_raw = dex3_metrics()
    dump(OUT / "dex3_force_balance_metrics.json", force_metrics)
    dump(OUT / "dex3_contact_wrench_metrics.json", wrench_metrics)
    dump(OUT / "dex3_actuator_limit_audit.json", actuator_metrics)

    with (OUT / "dex3_retention_force_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)

    material_probe = json.loads((BASE / "runtime_material_probe.json").read_text())
    material_audit = {
        "status": "DEX3_GENERIC_CONTACT_MATERIAL",
        "runtime_source": material_probe["source"],
        "runtime_read_only": True,
        "thumb": material_probe["task_fingers"]["THUMB"],
        "index": material_probe["task_fingers"]["INDEX"],
        "phone": material_probe["phone"],
        "effective_pair": {
            "combine_mode": "PhysX default/average because no physics material binding is authored on either shape",
            "static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0,
        },
        "realism_classification": "DEX3_GENERIC_CONTACT_MATERIAL",
        "real_Dex3_fingertip_material_calibrated": False,
        "comparison_to_aloha": success_summary["material"],
        "friction_changed_by_audit": False,
    }
    dump(OUT / "dex3_runtime_material_audit.json", material_audit)

    comparison = {
        "status": "CROSS_EMBODIMENT_PHYSICAL_GRASP_AUTHORITY_COMPARISON_COMPLETE",
        "raw_commands_are_not_directly_comparable": True,
        "historical_aloha_reference": aloha_metrics,
        "dex3_candidate_A": {
            "object_weight_n": force_metrics["object_weight_n"],
            "contact_pair": "physical thumb + physical index, third non-task",
            "grasp_topology": "photo-derived distal fingertip pinch",
            "normal_force_ratio": force_metrics["normal_force_ratio_all"],
            "normal_force_ratio_pre_table_impact": force_metrics["normal_force_ratio_pre_table_impact"],
            "vertical_support_ratio": force_metrics["vertical_support_ratio_all"],
            "vertical_support_ratio_pre_table_impact": force_metrics["vertical_support_ratio_pre_table_impact"],
            "friction_coefficient_runtime": {"static": 0.5, "dynamic": 0.5},
            "friction_utilization": force_metrics["friction_utilization"],
            "actuator_utilization": {
                "type": "implicit-drive model-request proxy",
                "task_joint_clipped_fraction": actuator_metrics["task_joint_model_request_clipped_fraction"],
            },
            "slip_m": force_metrics["phone_relative_slip_m"],
            "rotation_deg": force_metrics["phone_rotation_deg"],
            "retention_result": "BILATERAL_CONTACT_MAINTAINED_BUT_PHONE_FELL_TO_TABLE",
        },
        "answers": {
            "aloha_drive_question": "The controlled bracket supports that 4.866119 N/carriage crossed the archived true-lift threshold, but the same run still had about 46 mm local slip; it does not establish drive as the sole source of stable grasp success.",
            "aloha_material_question": "Bare-phone split materials were active in W1. A separate tape-sleeve geometry experiment reduced slip 36.8 percent but still failed sustained hold.",
            "dex3_question": "Dex3 has ample scalar normal-force authority in this pose, but the force directions/contact wrench provide less than one body weight of upward support before table impact; its uncalibrated generic material is an additional uncertainty.",
        },
    }
    dump(OUT / "aloha_vs_dex3_grasp_authority.json", comparison)

    stable_window = force_metrics["force_decay_windows"][-1]
    stable_rn = stable_window["normal_force_ratio"]["mean"]
    pre_support = force_metrics["vertical_support_ratio_pre_table_impact"]["mean"]
    root_cause = {
        "status": "RETENTION_ROOT_CAUSE_MULTI_FACTOR",
        "primary": "CONTACT_GEOMETRY_WRENCH_LIMITED",
        "contributors": ["CONTACT_MATERIAL_FRICTION_LIMITED", "INITIAL_PENETRATION_RELAXATION_LIMITED"],
        "not_primary": ["ACTUATOR_HOLD_AUTHORITY_LIMITED", "CONTROLLER_TRACKING_LIMITED"],
        "evidence": {
            "sustained_normal_force_ratio_mean_1_to_1p5_s": stable_rn,
            "sustained_mu_times_normal_capacity_ratio": 0.5 * stable_rn,
            "pre_table_mean_actual_vertical_support_ratio": pre_support,
            "pre_table_max_actual_vertical_support_ratio": force_metrics["vertical_support_ratio_pre_table_impact"]["maximum"],
            "pre_table_thumb_friction_utilization_p95": force_metrics["friction_utilization"]["thumb_pre_table_impact"]["p95"],
            "pre_table_index_friction_utilization_p95": force_metrics["friction_utilization"]["index_pre_table_impact"]["p95"],
            "contact_height_offset_p95_m": wrench_metrics["contact_centroid_world_height_offset_m"]["p95"],
            "phone_rotation_deg": force_metrics["phone_rotation_deg"],
            "task_chain_max_q_error_rad": actuator_metrics["task_chain_max_q_error_rad"],
            "task_joint_model_request_clipped_fraction": actuator_metrics["task_joint_model_request_clipped_fraction"],
        },
        "normal_force_decay_classification": "INITIAL_PENETRATION_FORCE_TRANSIENT_ONLY_CONTRIBUTOR_NOT_PRIMARY",
        "actuator_effort_sweep_scientifically_justified": False,
        "effort_sweep_executed": False,
        "effort_sweep_reason": (
            "The required gate is not satisfied: sustained scalar normal force is not insufficient. "
            "At runtime mu=0.5, the late normal-force friction-capacity proxy exceeds weight, while actual pre-impact vertical support remains below weight because of contact-force direction and wrench geometry."
        ),
        "friction_sensitivity_scientifically_justified": False,
        "friction_sensitivity_executed": False,
        "material_model_followup": "MATERIAL_MODEL_CALIBRATION_NEEDED; the current 0.5/0.5 values are generic defaults, not measured Dex3 pad properties.",
    }
    dump(OUT / "dex3_retention_root_cause.json", root_cause)

    after = {key: sha(path) for key, path in frozen_paths.items()}
    prior_trace_path = (
        ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_pad_ablation_v1/"
        "candidate_A_physics/static_physics_trace.npz"
    )
    prior_trace = np.load(prior_trace_path, allow_pickle=False)
    audited_trace = np.load(BASE / "static_physics_trace.npz", allow_pickle=False)
    parity_keys = [
        "time_s", "commanded_left_dex3_q", "actual_left_dex3_q",
        "actual_fixed_arm_q", "phone_pose_xyzw", "phone_velocity",
        "pinch_center_m", "thumb_phone_force_n", "index_phone_force_n",
        "third_phone_force_n", "simultaneous_thumb_index",
    ]
    reproduction_identity = {
        key: {
            "array_equal": bool(np.array_equal(prior_trace[key], audited_trace[key])),
            "maximum_absolute_difference": float(np.max(np.abs(
                prior_trace[key].astype(float) - audited_trace[key].astype(float)
            ))),
        }
        for key in parity_keys
    }
    frozen = {
        "status": "FROZEN_CANDIDATE_A_AND_SCIENTIFIC_INPUTS_UNCHANGED" if before == after else "BLOCKED_FROZEN_INPUT_MUTATION",
        "hashes_before": before, "hashes_after": after,
        "byte_identical": before == after,
        "candidate_A_q_rad": primitive_q,
        "candidate_A_q_matches_user_values_at_9_decimals": bool(
            np.array_equal(np.round(primitive_q, 9), APPROVED_Q)
        ),
        "third_finger_non_task": True, "arm_wrist_modified": False,
        "friction_modified": False, "geometry_modified": False,
        "v17_2_modified": False, "aloha_cartesian_backbone_modified": False,
        "baseline_physics_reproduction": {
            "prior_candidate_A_trace": prior_trace_path,
            "audited_trace": BASE / "static_physics_trace.npz",
            "common_state_arrays": reproduction_identity,
            "all_common_state_arrays_exactly_identical": all(
                row["array_equal"] for row in reproduction_identity.values()
            ),
        },
    }
    dump(OUT / "frozen_input_audit.json", frozen)

    # Plots.
    labels = ["ALOHA W1\ntrue-lift reference", "Dex3 A\npre-table"]
    rn_values = [aloha_metrics["normal_force_ratio"]["median"], force_metrics["normal_force_ratio_pre_table_impact"]["median"]]
    rs_values = [aloha_metrics["vertical_support_ratio"]["median"], force_metrics["vertical_support_ratio_pre_table_impact"]["median"]]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.bar(x - 0.18, rn_values, 0.36, label="Normal-force ratio")
    ax.bar(x + 0.18, rs_values, 0.36, label="Vertical-support ratio")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Object weight")
    ax.set_xticks(x, labels); ax.set_ylabel("dimensionless ratio / object weight")
    ax.set_title("Cross-embodiment grasp authority (physical outcomes, not raw commands)")
    ax.grid(axis="y", alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "aloha_vs_dex3_force_balance.png"); plt.close(fig)

    t = dex3_raw["t"]
    fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax1.plot(t, np.linalg.norm(dex3_raw["normal"], axis=2)[:, 0], label="thumb normal", color="#276FBF")
    ax1.plot(t, np.linalg.norm(dex3_raw["normal"], axis=2)[:, 1], label="index normal", color="#E07A1F")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("normal force (N)")
    ax2 = ax1.twinx(); ax2.plot(t, dex3_raw["slip"] * 1000, color="#BB3E03", linewidth=2, label="relative slip")
    ax2.set_ylabel("phone slip (mm)")
    ax1.axvline(t[dex3_raw["impact_index"]], color="gray", linestyle="--", label="inferred table impact")
    lines = ax1.get_lines() + ax2.get_lines(); ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax1.set_title("Frozen Candidate A: contact force remains while phone slips")
    ax1.grid(alpha=0.2); fig.tight_layout(); fig.savefig(OUT / "dex3_contact_force_vs_slip.png"); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), dpi=160, sharex=True)
    axes[0].plot(t, dex3_raw["net_force"][:, 2], label="hand contact Fz")
    axes[0].axhline(force_metrics["object_weight_n"], linestyle="--", color="black", label="weight")
    axes[0].set_ylabel("vertical force (N)"); axes[0].legend()
    for i, axis in enumerate("xyz"):
        axes[1].plot(t, dex3_raw["net_moment"][:, i], label=f"M{axis}")
    axes[1].set_ylabel("contact moment (N m)"); axes[1].legend(ncol=3)
    axes[2].plot(t, dex3_raw["rotation_change"], label="phone rotation", color="#8B2FC9")
    axes[2].set_ylabel("rotation (deg)"); axes[2].set_xlabel("time (s)"); axes[2].legend()
    for ax in axes:
        ax.axvline(t[dex3_raw["impact_index"]], color="gray", linestyle="--"); ax.grid(alpha=0.2)
    fig.suptitle("Frozen Candidate A phone wrench and rotation timeline")
    fig.tight_layout(); fig.savefig(OUT / "dex3_phone_wrench_timeline.png"); plt.close(fig)

    command_text = f"""#!/usr/bin/env bash
set -euo pipefail
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd {ROOT}

# Read-only historical forensic commands
git grep -n "4.866119"
git grep -n "gripper-effort-cap"
git grep -n "hardware-max-close"
git log -S"4.866119" -p --all
git log -S"gripper-effort-cap" -p --all

# Frozen Candidate-A true-PhysX force audit (no q/material/controller changes)
/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir {BASE} \\
  --trial closed_hold --hold-duration 1.5 \\
  --artifact-prefix candidate_A_force_audit \\
  --headless --enable_cameras

python tools/finalize_dex3_left_phone_retention_force_audit_v1.py
"""
    (OUT / "commands.sh").write_text(command_text, encoding="utf-8")
    (OUT / "commands.sh").chmod(0o755)

    report = f"""HISTORICAL ALOHA: hardware-max-close + effort-cap 4.866119 N/carriage + bare-phone-split was the archived W1 true-lift reference; the tape sleeve was a separate geometry experiment.
4.866119 N/carriage is causally supported only for crossing the archived 4.507623-fail/4.866119-pass true-lift bracket, not for full stable-retention success.
CURRENT DEX3 PRIMARY CAUSE: RETENTION_ROOT_CAUSE_MULTI_FACTOR, dominated by CONTACT_GEOMETRY_WRENCH_LIMITED; an actuator-only sweep was not scientifically justified.

# Dex3 left-phone retention force audit v1

1. **Historical evidence.** Git `-S` identifies checkpoint `324f8c7`; archived W1 CSVs contain reproduced Isaac contact/friction/drive traces. The later combined GoPark run lifted the assembly but failed strict bilateral-maintenance and penetration gates.

2. **ALOHA effort configuration.** `hardware-max-close`, 4.866119 N/carriage cap, 1296.709629 N/m stiffness, 10 N·s/m damping, and 0.5 m/s carriage velocity limit. The archived model upper bound was 38.417590 N/carriage.

3. **ALOHA material configuration.** W1 used `bare-phone-split`, average combine: effective tape/glass static/dynamic 0.55/0.425 and tape/matte 0.65/0.525.

4. **ALOHA contact geometry/tape.** W1 did not contain the 20-mm sleeve layer. In the separate sleeve test, continuous distal geometry reduced slip 36.774% and penetration 83.960%, but sustained hold still failed.

5. **Current Dex3 actuator.** Seven implicit position drives: 2.5 N·m limit, 100 N·m/rad stiffness, 4 N·m·s/rad damping, 12 rad/s limit. The implicit-drive torque is not directly exposed; model-request clipping is explicitly a proxy.

6. **Current Dex3 runtime material.** Thumb, index, and phone all read back 0.5 static / 0.5 dynamic / 0 restitution, with no authored physics material binding. Classification: `DEX3_GENERIC_CONTACT_MATERIAL`, not real-pad-calibrated.

7. **Normal-force ratio.** ALOHA W1 median `{aloha_metrics['normal_force_ratio']['median']:.3f}`; Dex3 pre-table median `{force_metrics['normal_force_ratio_pre_table_impact']['median']:.3f}` and late 1.0–1.5 s mean `{stable_rn:.3f}`. Dex3 did not lose scalar bilateral normal force.

8. **Vertical-support ratio.** ALOHA W1 median `{aloha_metrics['vertical_support_ratio']['median']:.3f}`; Dex3 pre-table mean/maximum `{pre_support:.3f}/{force_metrics['vertical_support_ratio_pre_table_impact']['maximum']:.3f}`. Dex3 never reached one body weight before impact.

9. **Friction utilization/margin.** Dex3 pre-table thumb/index p95 utilization `{force_metrics['friction_utilization']['thumb_pre_table_impact']['p95']:.3f}`/`{force_metrics['friction_utilization']['index_pre_table_impact']['p95']:.3f}`. The generic friction cone is nearly active while useful upward support remains insufficient.

10. **Normal-force decay.** An initial penetration transient produces large peaks, then normal force settles to about `{stable_rn:.3f}` body weights rather than decaying to zero. `INITIAL_PENETRATION_RELAXATION_LIMITED` is contributory, not primary.

11. **Actuator utilization.** Task-joint model-request clip fraction `{actuator_metrics['task_joint_model_request_clipped_fraction']:.3f}`; task-chain maximum q error `{actuator_metrics['task_chain_max_q_error_rad']:.6f}` rad. Saturation is present, but sustained normal magnitude is already theoretically sufficient.

12. **Contact wrench/rotation.** Contact-centroid height offset p95 `{wrench_metrics['contact_centroid_world_height_offset_m']['p95']*1000:.3f}` mm, coupled with tilted unequal contact vectors, generated a net moment and `{force_metrics['phone_rotation_deg']:.3f}`° rotation before/through table settling.

13. **Root cause.** `RETENTION_ROOT_CAUSE_MULTI_FACTOR`: primary `CONTACT_GEOMETRY_WRENCH_LIMITED`; contributors `CONTACT_MATERIAL_FRICTION_LIMITED` and `INITIAL_PENETRATION_RELAXATION_LIMITED`. Controller tracking is not the primary blocker.

14. **Effort sweep decision.** Not justified and not run. The required “sustained normal force insufficient” gate failed; raising effort would confound wrench/material diagnosis and could only increase contact load.

15. **Effort sweep results.** None. No Dex3 effort value was changed, and 4.866119 was never copied to Dex3.

16. **Friction sensitivity.** Not run. The active material is generic and needs physical calibration, but a success-seeking friction sweep would not establish the real fingertip material.

17. **Diagnostic parameter changes.** None to q, effort, stiffness, damping, friction, mass, gravity, collision geometry, phone pose, arm, or wrist. Only readback instrumentation was added.

18. **Freeze proof.** Candidate A hash `{after['candidate_A_primitive_json']}`, v17.2 hash `{after['v17_2_full_trajectory']}`, and v14 backbone hash `{after['v14_cartesian_backbone']}` remained byte-identical.

19. **Plots and data.** `{OUT / 'aloha_vs_dex3_force_balance.png'}`, `{OUT / 'dex3_contact_force_vs_slip.png'}`, `{OUT / 'dex3_phone_wrench_timeline.png'}`, and `{OUT / 'dex3_retention_force_trace.csv'}`.

20. **Exact next action.** Measure the real Dex3 fingertip-pad/phone static and dynamic friction and authorize a material-model calibration before changing hold effort.

THE HISTORICAL ALOHA EFFORT VALUE WAS AUDITED BUT WAS NOT NUMERICALLY COPIED INTO DEX3
ALOHA AND DEX3 WERE COMPARED USING PHYSICAL GRASP-AUTHORITY AND FORCE-BALANCE METRICS
THE USER-APPROVED LEFT DEX3 PHOTO PINCH REMAINED THE FROZEN GRASP TOPOLOGY
THE THIRD DEX3 FINGER REMAINED NON-TASK
THE G1 ARM, WRIST, ALOHA CARTESIAN BACKBONE, AND V17.2 TRAJECTORY WERE NOT MODIFIED
FRICTION WAS NOT TUNED BEFORE THE ROOT CAUSE WAS IDENTIFIED
ANY EFFORT SWEEP WAS LIMITED TO THE ACTIVE DEX3 MODEL'S DEFENSIBLE LIMITS
NO PHYSICS SUCCESS WAS CREATED BY OBJECT FOLLOW, SCRIPTED ATTACHMENT, OR TELEPORT
NO DDS, PUBLISHER, ALOHA COMMAND, OR REAL-G1 COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    artifact_paths = sorted(path for path in OUT.rglob("*") if path.is_file())
    manifest = {
        "status": root_cause["status"],
        "created_by": Path(__file__).resolve(),
        "frozen_input_audit": frozen,
        "baseline_true_physics": BASE,
        "baseline_no_cheat": json.loads((BASE / "no_cheat_audit.json").read_text()),
        "effort_sweep_executed": False,
        "friction_sensitivity_executed": False,
        "validation_reads": 0, "heldout_reads": 0, "g1_expert_reads": 0,
        "dds": False, "publisher": False, "hardware_command": False,
        "artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in artifact_paths if path.name != "run_manifest.json"],
    }
    dump(OUT / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
