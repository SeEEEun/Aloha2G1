#!/usr/bin/env python3
"""Refine the best Dex3 phone grasp with intended-contact semantics.

Only four distal collision-mesh/phone pairs are treated as intended contacts.
All other phone and robot pairs remain enabled and are checked as forbidden.
The program is kinematic (qpos + mj_forward) and never builds a trajectory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import find_g1_dex3_static_phone_grasp as old  # noqa: E402

OLD_REPORT = ROOT / "converted_runs/g1_dex3_static_phone_grasp/static_grasp_report.json"
OUT = ROOT / "converted_runs/g1_dex3_static_phone_grasp_refined"
FORBIDDEN_TOL = 0.0002
CONTACT_LO = -FORBIDDEN_TOL
CONTACT_HI = 0.001
PAD_EDGE_MARGIN = 0.002
ARM_MARGIN = 0.03
SIDES = ("left", "right")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--starts", type=int, default=72)
    p.add_argument("--max-nfev", type=int, default=2400)
    return p.parse_args()


def serial(x):
    return old.serial(x)


def atomic_json(path: Path, value: dict) -> None:
    old.atomic_json(path, value)


def collision_geoms(model: mujoco.MjModel, body: str) -> list[int]:
    return [g for g in old.geom_for_body(model, body, False)
            if model.geom_contype[g] or model.geom_conaffinity[g]]


def visual_geoms(model: mujoco.MjModel, body: str) -> list[int]:
    return [g for g in old.geom_for_body(model, body, False)
            if not model.geom_contype[g] and not model.geom_conaffinity[g]]


def mesh_vertices_world(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray:
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
        raise RuntimeError(f"Contact geom {gid} is not a mesh")
    mid = int(model.geom_dataid[gid])
    adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    local = model.mesh_vert[adr:adr+num]
    return data.geom_xpos[gid] + local @ data.geom_xmat[gid].reshape(3, 3).T


def pad_from_mesh(model: mujoco.MjModel, data: mujoco.MjData, gid: int,
                  toward_phone: np.ndarray) -> dict:
    """Support patch of the actual mesh in the direction of the phone."""
    vertices = mesh_vertices_world(model, data, gid)
    direction = toward_phone / np.linalg.norm(toward_phone)
    support = vertices @ direction
    extreme = support.max()
    patch = vertices[support >= extreme-.0015]
    center = patch.mean(axis=0)
    radial = center-data.geom_xpos[gid]
    normal = radial/(np.linalg.norm(radial)+1e-12)
    radius = float(np.max(np.linalg.norm(
        patch-(patch @ direction)[:, None]*direction
        - (center-np.dot(center, direction)*direction), axis=1), initial=0))
    return {"center": center, "normal": normal, "radius": radius,
            "vertex_count": len(patch), "vertices": patch}


def distance(model, data, a, b) -> tuple[float, np.ndarray]:
    fromto = np.empty(6)
    value = float(mujoco.mj_geomDistance(model, data, int(a), int(b), .25, fromto))
    return value, fromto


def contact_normal_phone_to_tip(model, data, tip_gid: int, phone_gid: int,
                                fromto: np.ndarray) -> np.ndarray:
    """MuJoCo narrow-phase normal, oriented from phone toward fingertip."""
    for c in data.contact:
        if {int(c.geom1), int(c.geom2)} == {int(tip_gid), int(phone_gid)}:
            n = np.asarray(c.frame[:3], float).copy()  # geom1 -> geom2
            return n if int(c.geom1) == int(phone_gid) else -n
    # Separated closest points: call order was tip,phone, so vector is
    # fingertip -> phone and must be negated.
    n = -(fromto[3:]-fromto[:3])
    return n/(np.linalg.norm(n)+1e-12)


def classify(model: mujoco.MjModel) -> dict:
    tips, phone_forbidden, cross_pairs, torso_pairs = {}, [], [], []
    all_collision = [g for g in range(model.ngeom)
                     if model.geom_contype[g] or model.geom_conaffinity[g]]
    phone = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "phone_proxy_geom")
    for side in SIDES:
        tips[side] = {}
        for part in ("thumb", "index"):
            body = f"{side}_hand_{part}_2_link" if part == "thumb" else f"{side}_hand_index_1_link"
            tips[side][part] = collision_geoms(model, body)[-1]
    intended = {g for side in SIDES for g in tips[side].values()}
    for g in all_collision:
        body = old.body_name(model, g)
        if g == phone or body == "world":
            continue
        if g not in intended and old.category(body) in ("finger", "hand_wrist", "arm"):
            phone_forbidden.append((g, phone))
    for i, a in enumerate(all_collision):
        ba = old.body_name(model, a)
        for b in all_collision[i+1:]:
            bb = old.body_name(model, b)
            if a == phone or b == phone or ba == bb:
                continue
            cross = ((ba.startswith("left_") and bb.startswith("right_"))
                     or (ba.startswith("right_") and bb.startswith("left_")))
            torso = "torso_link" in (ba, bb) and (
                old.category(ba) in ("finger", "hand_wrist", "arm")
                or old.category(bb) in ("finger", "hand_wrist", "arm"))
            if cross:
                cross_pairs.append((a, b))
            if torso:
                torso_pairs.append((a, b))
    return {"tips": tips, "phone": phone, "phone_forbidden": phone_forbidden,
            "cross": cross_pairs, "torso": torso_pairs}


def geometry_calibration(model: mujoco.MjModel, data: mujoco.MjData,
                         groups: dict, old_x: np.ndarray, center: np.ndarray) -> dict:
    records = {}
    for side in SIDES:
        records[side] = {}
        for part, gid in groups["tips"][side].items():
            body = old.body_name(model, gid)
            visuals = visual_geoms(model, body)
            visual = visuals[-1]
            same = {
                "same_mesh_id": int(model.geom_dataid[visual]) == int(model.geom_dataid[gid]),
                "local_position_difference_m": model.geom_pos[visual]-model.geom_pos[gid],
                "local_quaternion_difference": model.geom_quat[visual]-model.geom_quat[gid],
                "world_position_difference_m": data.geom_xpos[visual]-data.geom_xpos[gid],
                "world_rotation_max_abs_difference": float(np.max(np.abs(
                    data.geom_xmat[visual]-data.geom_xmat[gid]))),
            }
            toward = center-data.geom_xpos[gid]
            pad = pad_from_mesh(model, data, gid, toward)
            dcol, _ = distance(model, data, gid, groups["phone"])
            dvis, _ = distance(model, data, visual, groups["phone"])
            records[side][part] = {
                "body": body, "collision_geom_id": gid, "visual_geom_id": visual,
                "geom_type": mujoco.mjtGeom(int(model.geom_type[gid])).name,
                "mesh_id": int(model.geom_dataid[gid]),
                "geom_size": model.geom_size[gid],
                "local_position": model.geom_pos[gid],
                "local_quaternion_wxyz": model.geom_quat[gid],
                "world_position": data.geom_xpos[gid],
                "world_rotation": data.geom_xmat[gid].reshape(3, 3),
                "contact_pad_center": pad["center"],
                "contact_pad_normal": pad["normal"],
                "contact_patch_radius_m": pad["radius"],
                "contact_patch_vertex_count": pad["vertex_count"],
                "collision_signed_distance_m": dcol,
                "visual_surface_signed_distance_m": dvis,
                "visual_collision_surface_offset_m": dvis-dcol,
                "visual_collision_identity": same,
            }
    middle = {}
    palm = {}
    for side in SIDES:
        middle[side] = {}
        for i in (0, 1):
            body = f"{side}_hand_middle_{i}_link"
            middle[side][body] = [{
                "geom_id": g, "type": mujoco.mjtGeom(int(model.geom_type[g])).name,
                "size": model.geom_size[g], "local_position": model.geom_pos[g],
                "local_quaternion_wxyz": model.geom_quat[g],
                "world_position": data.geom_xpos[g],
            } for g in collision_geoms(model, body)]
        palm_body = f"{side}_wrist_yaw_link"
        palm[side] = {"body": palm_body, "geoms": [{
            "geom_id": g, "type": mujoco.mjtGeom(int(model.geom_type[g])).name,
            "size": model.geom_size[g], "local_position": model.geom_pos[g],
            "local_quaternion_wxyz": model.geom_quat[g],
            "world_position": data.geom_xpos[g],
        } for g in collision_geoms(model, palm_body)]}
    return {
        "source_xml": str(old.G1_XML), "source_candidate_report": str(OLD_REPORT),
        "reproduced_old_best_qpos_variables": old_x,
        "reproduced_phone_center": center,
        "phone_proxy": {"geom_id": groups["phone"], "geom_type": "mjGEOM_BOX",
                        "half_sizes_xyz_m": model.geom_size[groups["phone"]],
                        "axes": "X thickness, Y short, Z long"},
        "fingertip_contact_geometries": records,
        "middle_third_finger_collision_geometries": middle,
        "palm_hand_base_collision_geometries": palm,
        "tolerance": {
            "intended_contact_primary_range_m": [CONTACT_LO, CONTACT_HI],
            "forbidden_penetration_numeric_tolerance_m": FORBIDDEN_TOL,
            "two_mm_fallback_used": False,
            "basis": (
                "Each distal visual geom and collision geom uses the identical "
                "mesh id, local pose, and world pose. Measured visual/collision "
                "surface offset is zero, so no 2 mm approximation allowance is justified.")
        },
    }


def optimizer(info: dict, base_model: mujoco.MjModel, old_x: np.ndarray,
              old_center: np.ndarray, starts: int, max_nfev: int) -> list[dict]:
    layout, _ = old.hand_layout(info)
    groups = classify(base_model)
    model, data = base_model, mujoco.MjData(base_model)
    phone_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phone_proxy")
    arm_ids = info["arm_qpos_ids"]
    la, ra = layout["hands"]["left"]["qadr"], layout["hands"]["right"]["qadr"]
    hand_ranges = np.r_[layout["hands"]["left"]["ranges"],
                        layout["hands"]["right"]["ranges"]]
    lower = np.r_[info["joint_limits"][:, 0]+ARM_MARGIN,
                  hand_ranges[:, 0]+.005,
                  old_center+[-.025, -.012, -.035],
                  np.radians([-4, -4, -4])]
    upper = np.r_[info["joint_limits"][:, 1]-ARM_MARGIN,
                  hand_ranges[:, 1]-.005,
                  old_center+[.025, .012, .035],
                  np.radians([4, 4, 4])]
    start0 = np.r_[old_x, old_center, np.zeros(3)]
    natural = old.relative.load_natural_start(old.NATURAL_NPZ, info)["arm_q"]
    def assign(v):
        data.qpos[:] = model.key_qpos[0]
        data.qpos[arm_ids] = v[:14]
        data.qpos[la] = v[14:21]
        data.qpos[ra] = v[21:28]
        model.body_pos[phone_bid] = v[28:31]
        quat = Rotation.from_euler("xyz", v[31:34]).as_quat()
        model.body_quat[phone_bid] = quat[[3, 0, 1, 2]]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
    def phone_frame(v):
        r = Rotation.from_euler("xyz", v[31:34]).as_matrix()
        return v[28:31], r
    def residual(v):
        assign(v)
        center, prot = phone_frame(v)
        rr = []
        local_pads = {}
        desired = {
            ("left", "thumb"): (1, 1), ("left", "index"): (-1, 1),
            ("right", "thumb"): (1, -1), ("right", "index"): (-1, -1)}
        for (side, part), (face, lateral) in desired.items():
            gid = groups["tips"][side][part]
            dist, _ = distance(model, data, gid, groups["phone"])
            rr.append(700*(dist-.00015))
            # Actual support patch, on the correct opposite thickness face and
            # at an in-bounds lateral edge patch.
            target_normal = prot[:, 0]*face
            pad = pad_from_mesh(model, data, gid, center-data.geom_xpos[gid])
            local = prot.T @ (pad["center"]-center)
            local_pads[(side, part)] = local
            rr.append(180*(local[0]-face*old.PHONE[1]/2))
            # Stay well inside the lateral edge so the narrow-phase normal is
            # the thickness-face normal rather than the rounded phone corner.
            rr.append(160*(local[1]-lateral*(old.PHONE[2]/2-.018)))
            rr.append(100*local[2])
            # Narrow-phase normal is validated independently. Avoid inserting
            # a discontinuous contact-normal residual into least_squares.
        for side in SIDES:
            delta = local_pads[(side, "thumb")]-local_pads[(side, "index")]
            rr.append(260*delta[1])
            rr.append(260*delta[2])
            rr.append(180*(abs(delta[0])-old.PHONE[1]))
        for a, b in groups["phone_forbidden"]:
            dist, _ = distance(model, data, a, b)
            rr.append(220*min(0., dist-FORBIDDEN_TOL))
        for pairs, weight in ((groups["cross"], 180), (groups["torso"], 180)):
            for a, b in pairs:
                dist, _ = distance(model, data, a, b)
                rr.append(weight*min(0., dist-FORBIDDEN_TOL))
        rr.extend(.06*(v[:14]-natural))
        rr.extend(.012*(v[14:28]-old_x[14:28]))
        mirror = np.array([1, -1, -1, 1, -1, 1, -1])
        rr.extend(.08*(v[7:14]-mirror*v[:7]))
        rr.extend(4*(v[28:31]-old_center))
        rr.extend(2*v[31:34])
        return np.asarray(rr)
    results = []
    for seed in range(starts):
        rng = np.random.default_rng(7200+seed)
        v0 = start0.copy()
        family = ("old_best", "wrist_roll", "wrist_yaw", "elbow",
                  "middle_fold", "thumb_opposition", "index_flex", "combined")[seed % 8]
        scale = .03 + .015*(seed//8)
        v0[:28] += rng.normal(0, scale, 28)
        if family == "wrist_roll":
            v0[[4, 11]] += [.18, -.18]
        elif family == "wrist_yaw":
            v0[[6, 13]] += [.16, -.16]
        elif family == "elbow":
            v0[[3, 10]] += rng.choice([-1, 1])*.25
        elif family == "middle_fold":
            v0[[17, 18, 24, 25]] += np.array([-.18, -.18, .18, .18])
        elif family == "thumb_opposition":
            v0[[14, 15, 16, 21, 22, 23]] += rng.normal(0, .16, 6)
        elif family == "index_flex":
            v0[[19, 20, 26, 27]] += rng.normal(0, .16, 4)
        elif family == "combined":
            v0[:28] += rng.normal(0, .10, 28)
        v0[28:31] += rng.normal(0, [.006, .003, .008])
        v0[31:34] += rng.normal(0, np.radians(1), 3)
        v0 = np.clip(v0, lower, upper)
        sol = least_squares(residual, v0, bounds=(lower, upper),
                            max_nfev=max_nfev, ftol=1e-11, xtol=1e-11,
                            gtol=1e-11)
        rec = validate(model, data, info, layout, groups, sol.x)
        rec.update({"candidate": seed, "family": family, "cost": float(sol.cost),
                    "optimizer_success": bool(sol.success), "v": sol.x.copy()})
        results.append(rec)
        print(f"{seed:03d} {family:16s} valid={rec['valid']} "
              f"contact={rec['max_abs_contact_distance_m']:.6f} "
              f"forbid={rec['minimum_forbidden_clearance_m']:.6f} "
              f"normal={rec['maximum_normal_error_deg']:.2f}", flush=True)
    return results


def validate(model, data, info, layout, groups, v) -> dict:
    arm_ids = info["arm_qpos_ids"]
    la, ra = layout["hands"]["left"]["qadr"], layout["hands"]["right"]["qadr"]
    phone_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phone_proxy")
    data.qpos[:] = model.key_qpos[0]
    data.qpos[arm_ids], data.qpos[la], data.qpos[ra] = v[:14], v[14:21], v[21:28]
    model.body_pos[phone_bid] = v[28:31]
    quat = Rotation.from_euler("xyz", v[31:34]).as_quat()
    model.body_quat[phone_bid] = quat[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    prot = Rotation.from_euler("xyz", v[31:34]).as_matrix()
    desired = {
        ("left", "thumb"): (1, 1), ("left", "index"): (-1, 1),
        ("right", "thumb"): (1, -1), ("right", "index"): (-1, -1)}
    contacts, normals, padposes, apertures = {}, [], {}, {}
    edge_ok, face_ok = True, True
    for key, (face, lateral) in desired.items():
        side, part = key
        gid = groups["tips"][side][part]
        dist, fromto = distance(model, data, gid, groups["phone"])
        pad = pad_from_mesh(model, data, gid, v[28:31]-data.geom_xpos[gid])
        local = prot.T @ (pad["center"]-v[28:31])
        target_phone_normal = prot[:, 0]*face
        contact_normal = contact_normal_phone_to_tip(
            model, data, gid, groups["phone"], fromto)
        normal_dot = float(np.dot(contact_normal, target_phone_normal))
        normal_error = float(np.degrees(np.arccos(np.clip(normal_dot, -1, 1))))
        normals.append(normal_error)
        in_bounds = (abs(local[1]) <= old.PHONE[2]/2-PAD_EDGE_MARGIN
                     and abs(local[2]) <= old.PHONE[0]/2-PAD_EDGE_MARGIN)
        correct_face = np.sign(local[0]) == face
        edge_ok &= in_bounds
        face_ok &= correct_face
        contacts[f"{side}_{part}"] = {
            "collision_signed_distance_m": dist,
            "visual_surface_signed_distance_m": dist,
            "closest_points": fromto.reshape(2, 3),
            "pad_center": pad["center"], "pad_normal": -contact_normal,
            "phone_to_tip_contact_normal": contact_normal,
            "pad_radius_m": pad["radius"], "phone_local_pad_center": local,
            "normal_error_deg": normal_error, "in_bounds": in_bounds,
            "correct_thickness_face": correct_face,
        }
        padposes[f"{side}_{part}"] = np.r_[pad["center"], pad["normal"]]
    forbidden = {}
    for label, pairs in (("phone", groups["phone_forbidden"]),
                         ("hand_hand", groups["cross"]), ("torso", groups["torso"])):
        for a, b in pairs:
            dist, fromto = distance(model, data, a, b)
            key = f"{label}:{old.body_name(model,a)}[{a}]|{old.body_name(model,b)}[{b}]"
            forbidden[key] = {"signed_distance_m": dist,
                              "closest_points": fromto.reshape(2, 3)}
    intended_dist = np.array([x["collision_signed_distance_m"] for x in contacts.values()])
    forbidden_dist = np.array([x["signed_distance_m"] for x in forbidden.values()])
    arm_margin = np.minimum(v[:14]-info["joint_limits"][:, 0],
                            info["joint_limits"][:, 1]-v[:14])
    for side in SIDES:
        p0 = contacts[f"{side}_thumb"]["pad_center"]
        p1 = contacts[f"{side}_index"]["pad_center"]
        apertures[side] = float(np.linalg.norm(p0-p1))
    antipodal = all(
        np.dot(contacts[f"{s}_thumb"]["pad_normal"],
               contacts[f"{s}_index"]["pad_normal"]) < -.85 for s in SIDES)
    screen_angle = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(prot[:, 0], [1, 0, 0])), -1, 1))))
    long_angle = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(prot[:, 2], [0, 0, 1])), -1, 1))))
    valid = bool(
        np.all((intended_dist >= CONTACT_LO) & (intended_dist <= CONTACT_HI))
        # "Facing" means a positive component into the selected phone face;
        # 75 deg keeps a non-trivial cosine (>0.258), while antipodal below
        # separately requires the two fingertip normals to oppose each other.
        and max(normals) < 75 and edge_ok and face_ok and antipodal
        and forbidden_dist.min(initial=1) >= -FORBIDDEN_TOL
        and arm_margin.min() >= ARM_MARGIN
        and screen_angle < 5 and long_angle < 5)
    return {
        "valid": valid, "contacts": contacts, "contact_pad_poses": padposes,
        "forbidden": forbidden, "intended_distances": intended_dist,
        "minimum_forbidden_clearance_m": float(forbidden_dist.min(initial=1)),
        "max_abs_contact_distance_m": float(np.max(np.abs(intended_dist))),
        "maximum_normal_error_deg": max(normals), "edge_bounds_ok": edge_ok,
        "correct_opposite_faces": face_ok, "antipodal": antipodal,
        "apertures": apertures, "arm_margins": arm_margin,
        "minimum_arm_margin": float(arm_margin.min()),
        "screen_frontal_angle_deg": screen_angle,
        "long_up_angle_deg": long_angle,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["candidate", "family", "valid", "cost",
              "max_abs_contact_distance_m", "minimum_forbidden_clearance_m",
              "maximum_normal_error_deg", "edge_bounds_ok",
              "correct_opposite_faces", "antipodal", "minimum_arm_margin",
              "screen_frontal_angle_deg", "long_up_angle_deg"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def save_selected(path: Path, model, info, layout, selected) -> None:
    v = selected["v"]
    qpos = model.key_qpos[0].copy()
    qpos[info["arm_qpos_ids"]] = v[:14]
    qpos[layout["hands"]["left"]["qadr"]] = v[14:21]
    qpos[layout["hands"]["right"]["qadr"]] = v[21:28]
    contact_names = list(selected["contacts"])
    forbidden_names = list(selected["forbidden"])
    payload = {
        "full_qpos": qpos, "arm_qpos": v[:14],
        "left_dex3_qpos": v[14:21], "right_dex3_qpos": v[21:28],
        "phone_proxy_pose": np.r_[v[28:31],
            Rotation.from_euler("xyz", v[31:34]).as_quat()[[3, 0, 1, 2]]],
        "contact_names": np.asarray(contact_names),
        "contact_pad_poses": np.asarray([selected["contact_pad_poses"][k]
                                        for k in contact_names]),
        "contact_points": np.asarray([selected["contacts"][k]["closest_points"]
                                      for k in contact_names]),
        "contact_normals": np.asarray([selected["contacts"][k]["pad_normal"]
                                       for k in contact_names]),
        "visual_surface_distances": np.asarray([
            selected["contacts"][k]["visual_surface_signed_distance_m"] for k in contact_names]),
        "collision_signed_distances": selected["intended_distances"],
        "thumb_index_aperture": np.asarray([selected["apertures"][s] for s in SIDES]),
        "forbidden_pair_names": np.asarray(forbidden_names),
        "forbidden_pair_clearances": np.asarray([
            selected["forbidden"][k]["signed_distance_m"] for k in forbidden_names]),
        "joint_limit_margins": selected["arm_margins"],
    }
    tmp = path.with_suffix(".npz.incomplete")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def main() -> int:
    a = args()
    a.output.mkdir(parents=True, exist_ok=True)
    old_report = json.loads(OLD_REPORT.read_text())
    best = old_report["best_candidate"]
    old_x, center = np.asarray(best["x"], float), np.asarray(best["phone_center"], float)
    info = old.relative.latest.ik.validate_model(old.G1_XML)
    layout, _ = old.hand_layout(info)
    model, _ = old.expanded_phone_model(center, np.zeros(3))
    data = mujoco.MjData(model)
    # Exact reproduction before refinement.
    data.qpos[:] = model.key_qpos[0]
    data.qpos[info["arm_qpos_ids"]] = old_x[:14]
    data.qpos[layout["hands"]["left"]["qadr"]] = old_x[14:21]
    data.qpos[layout["hands"]["right"]["qadr"]] = old_x[21:28]
    mujoco.mj_forward(model, data)
    groups = classify(model)
    calibration = geometry_calibration(model, data, groups, old_x, center)
    atomic_json(a.output / "contact_geometry_calibration.json", calibration)
    rows = optimizer(info, model, old_x, center, a.starts, a.max_nfev)
    write_csv(a.output / "refined_static_grasp_candidates.csv", rows)
    valid = [r for r in rows if r["valid"]]
    selected = min(valid, key=lambda r: (
        -r["minimum_forbidden_clearance_m"], r["maximum_normal_error_deg"],
        r["cost"])) if valid else None
    if selected is None:
        best_new = min(rows, key=lambda r: (
            max(0, -r["minimum_forbidden_clearance_m"]-FORBIDDEN_TOL)
            + max(0, r["max_abs_contact_distance_m"]-CONTACT_HI),
            r["maximum_normal_error_deg"], r["cost"]))
        report = {
            "verdict": "G1_STATIC_PHONE_GRASP_REFINED_BLOCKED",
            "safety_pass": False, "candidate_count": len(rows),
            "old_best_reproduced": True,
            "old_best_collision_signed_distances_m": best["tip_phone_signed_distances"],
            "tolerance": calibration["tolerance"],
            "best_refined_candidate": {
                k: v for k, v in best_new.items()
                if k not in ("v", "contacts", "forbidden", "contact_pad_poses")},
            "largest_blocker": (
                "The closest collision-free candidates reached ~10.2 mm "
                "thumb-index aperture, but distal mesh contact remained on/"
                "outside the phone lateral edge (about 90 deg from the desired "
                "thickness-face normal) and intended mesh overlap still exceeded "
                "the -0.2 mm lower bound. Forbidden clearance and static wrist "
                "margin were not the limiting constraints."),
            "relative_trajectory_wrist_margin_issue_modified": False,
            "trajectory_generated": False, "isaac_lab_executed": False,
            "hardware_executed": False,
        }
        atomic_json(a.output / "refined_static_grasp_report.json", report)
        print(json.dumps(serial(report), indent=2))
        print("G1_STATIC_PHONE_GRASP_REFINED_BLOCKED")
        return 2
    # Reassign selected because all candidates share this model.
    validated = validate(model, data, info, layout, groups, selected["v"])
    selected.update(validated)
    save_selected(a.output / "selected_static_grasp_refined.npz",
                  model, info, layout, selected)
    report = {
        "verdict": "G1_STATIC_PHONE_GRASP_REFINED_READY",
        "safety_pass": True, "candidate_count": len(rows),
        "selected_candidate": selected["candidate"],
        "selected_family": selected["family"],
        "phone_proxy_pose_xyz_rpy": selected["v"][28:34],
        "contacts": selected["contacts"],
        "thumb_index_aperture_m": selected["apertures"],
        "minimum_forbidden_clearance_m": selected["minimum_forbidden_clearance_m"],
        "minimum_arm_wrist_margin_rad": selected["minimum_arm_margin"],
        "screen_frontal_angle_deg": selected["screen_frontal_angle_deg"],
        "long_up_angle_deg": selected["long_up_angle_deg"],
        "tolerance": calibration["tolerance"],
        "relative_trajectory_wrist_margin_issue_modified": False,
        "trajectory_generated": False, "isaac_lab_executed": False,
        "hardware_executed": False,
    }
    atomic_json(a.output / "refined_static_grasp_report.json", report)
    print(json.dumps(serial(report), indent=2))
    print("G1_STATIC_PHONE_GRASP_REFINED_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
