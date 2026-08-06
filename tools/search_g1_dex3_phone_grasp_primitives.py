#!/usr/bin/env python3
"""Search Dex3-compatible static phone grasp primitives (kinematics only).

The rejected pure thumb-index thickness pinch is intentionally absent.
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
import refine_g1_dex3_static_phone_contact as refined  # noqa: E402

OUT = ROOT / "converted_runs/g1_dex3_phone_grasp_primitives"
SOURCE_REPORT = ROOT / ("converted_runs/g1_dex3_static_phone_grasp_refined/"
                        "refined_static_grasp_report.json")
CALIBRATION = SOURCE_REPORT.with_name("contact_geometry_calibration.json")
TYPES = ("THREE_POINT_FACE_CLAMP", "THUMB_INDEX_WITH_BOTTOM_SUPPORT",
         "BIMANUAL_SIDE_SUPPORT")
TYPE_SEEDS = {name: 19001+1000*i for i, name in enumerate(TYPES)}
CONTACT_RANGE = (-.0002, .001)
FORBIDDEN_TOL = .0002
NORMAL_LIMIT_DEG = 55.
ARM_MARGIN = .03


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--candidates-per-type", type=int, default=104)
    p.add_argument("--max-nfev", type=int, default=350)
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def topology(kind: str) -> list[dict]:
    """Desired phone-local pad points and outward phone normals."""
    hx, hy, hz = old.PHONE[1]/2, old.PHONE[2]/2, old.PHONE[0]/2
    out = []
    if kind == "THREE_POINT_FACE_CLAMP":
        for side, sy in (("left", 1), ("right", -1)):
            out += [
                dict(side=side, part="thumb", point=[hx, sy*.017, .018], normal=[1, 0, 0],
                     role="front_face_clamp"),
                dict(side=side, part="index", point=[-hx, sy*.017, .028], normal=[-1, 0, 0],
                     role="rear_face_upper_clamp"),
                dict(side=side, part="middle", point=[-hx, sy*.017, -.022], normal=[-1, 0, 0],
                     role="rear_face_lower_clamp"),
            ]
    elif kind == "THUMB_INDEX_WITH_BOTTOM_SUPPORT":
        for side, sy in (("left", 1), ("right", -1)):
            out += [
                dict(side=side, part="thumb", point=[hx, sy*.018, .015], normal=[1, 0, 0],
                     role="front_face_clamp"),
                dict(side=side, part="index", point=[-hx, sy*.018, .015], normal=[-1, 0, 0],
                     role="rear_face_clamp"),
                dict(side=side, part="middle", point=[0, sy*.020, -hz], normal=[0, 0, -1],
                     role="gravity_bottom_support"),
            ]
    elif kind == "BIMANUAL_SIDE_SUPPORT":
        # The two hands jointly oppose across the short/lateral dimension.
        for side, sy in (("left", 1), ("right", -1)):
            out += [
                dict(side=side, part="index", point=[0, sy*hy, .026], normal=[0, sy, 0],
                     role="lateral_upper_support"),
                dict(side=side, part="middle", point=[0, sy*hy, -.026], normal=[0, sy, 0],
                     role="lateral_lower_support"),
            ]
    else:
        raise KeyError(kind)
    return out


def body_for(side, part):
    return f"{side}_hand_{part}_{2 if part == 'thumb' else 1}_link"


def make_groups(model, specs):
    phone = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "phone_proxy_geom")
    intended = {}
    for s in specs:
        intended[(s["side"], s["part"])] = refined.collision_geoms(
            model, body_for(s["side"], s["part"]))[-1]
    intended_gids = set(intended.values())
    collision = [g for g in range(model.ngeom)
                 if model.geom_contype[g] or model.geom_conaffinity[g]]
    phone_forbidden, robot_pairs = [], []
    for g in collision:
        if g == phone or g in intended_gids:
            continue
        if old.category(old.body_name(model, g)) in ("finger", "hand_wrist", "arm"):
            phone_forbidden.append((g, phone))
    for i, a in enumerate(collision):
        ba = old.body_name(model, a)
        for b in collision[i+1:]:
            bb = old.body_name(model, b)
            if phone in (a, b) or ba == bb:
                continue
            cross = ((ba.startswith("left_") and bb.startswith("right_"))
                     or (ba.startswith("right_") and bb.startswith("left_")))
            torso = "torso_link" in (ba, bb) and (
                old.category(ba) in ("finger", "hand_wrist", "arm")
                or old.category(bb) in ("finger", "hand_wrist", "arm"))
            if cross or torso:
                robot_pairs.append((a, b))
    return dict(phone=phone, intended=intended, phone_forbidden=phone_forbidden,
                robot_pairs=robot_pairs)


def assign(model, data, info, layout, phone_bid, v):
    data.qpos[:] = model.key_qpos[0]
    data.qpos[info["arm_qpos_ids"]] = v[:14]
    data.qpos[layout["hands"]["left"]["qadr"]] = v[14:21]
    data.qpos[layout["hands"]["right"]["qadr"]] = v[21:28]
    model.body_pos[phone_bid] = v[28:31]
    q = Rotation.from_euler("xyz", v[31:34]).as_quat()
    model.body_quat[phone_bid] = q[[3, 0, 1, 2]]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def stability_proxy(points, applied_normals):
    """Unit-normal wrench span proxy, not a frictional force-closure proof."""
    p = np.asarray(points)
    f = np.asarray(applied_normals)
    center = p.mean(axis=0)
    wrench = np.vstack((f.T, np.cross(p-center, f).T))
    singular = np.linalg.svd(wrench, compute_uv=False)
    distribution = float(np.mean(np.linalg.norm(p-center, axis=1)))
    opposing = 0.
    for i in range(len(f)):
        for j in range(i+1, len(f)):
            opposing = max(opposing, -float(np.dot(f[i], f[j])))
    gravity_support = float(max(0., np.max(f[:, 2], initial=0)))
    return {
        "method": "CONTACT_STABILITY_PROXY",
        "wrench_matrix_rank": int(np.linalg.matrix_rank(wrench, tol=1e-4)),
        "minimum_nonzero_wrench_singular_value": float(
            singular[singular > 1e-5].min(initial=0)),
        "contact_distribution_radius_m": distribution,
        "maximum_opposing_force_cosine": opposing,
        "gravity_support_component": gravity_support,
        "score": float(.25*np.linalg.matrix_rank(wrench, tol=1e-4)
                       + opposing + 4*distribution + gravity_support),
        "force_closure_claimed": False,
    }


def narrowphase_contact_point(data, gid, phone_gid, fromto):
    for c in data.contact:
        if {int(c.geom1), int(c.geom2)} == {int(gid), int(phone_gid)}:
            return np.asarray(c.pos, float).copy(), "mujoco_contact_pos"
    return .5*(fromto[:3]+fromto[3:]), "closest_point_midpoint"


def evaluate(model, data, info, layout, phone_bid, groups, specs, v):
    assign(model, data, info, layout, phone_bid, v)
    prot = Rotation.from_euler("xyz", v[31:34]).as_matrix()
    contacts, points, applied, errors = [], [], [], []
    inside = True
    for spec in specs:
        gid = groups["intended"][(spec["side"], spec["part"])]
        dist, fromto = refined.distance(model, data, gid, groups["phone"])
        normal = refined.contact_normal_phone_to_tip(
            model, data, gid, groups["phone"], fromto)
        desired = prot @ np.asarray(spec["normal"], float)
        error = float(np.degrees(np.arccos(np.clip(np.dot(normal, desired), -1, 1))))
        pad = refined.pad_from_mesh(model, data, gid, v[28:31]-data.geom_xpos[gid])
        contact_point, point_source = narrowphase_contact_point(
            data, gid, groups["phone"], fromto)
        local = prot.T @ (contact_point-v[28:31])
        # All faces/edges require coordinates within the finite phone rectangle.
        this_inside = (abs(local[0]) <= old.PHONE[1]/2+.0002
                       and abs(local[1]) <= old.PHONE[2]/2+.0002
                       and abs(local[2]) <= old.PHONE[0]/2+.0002)
        inside &= this_inside
        points.append(contact_point)
        applied.append(-normal)  # fingertip force toward phone
        errors.append(error)
        contacts.append({
            **spec, "geom_id": gid, "body": old.body_name(model, gid),
            "collision_signed_distance_m": dist, "contact_point": contact_point,
            "contact_point_source": point_source,
            "phone_to_fingertip_normal": normal, "applied_force_direction": -normal,
            "normal_error_deg": error, "phone_local_point": local,
            "inside_designated_face_or_edge": this_inside,
        })
    forbidden = {}
    for label, pairs in (("phone", groups["phone_forbidden"]),
                         ("robot", groups["robot_pairs"])):
        for a, b in pairs:
            d, ft = refined.distance(model, data, a, b)
            forbidden[f"{label}:{old.body_name(model,a)}[{a}]|{old.body_name(model,b)}[{b}]"] = {
                "signed_distance_m": d, "closest_points": ft.reshape(2, 3)}
    fd = np.asarray([x["signed_distance_m"] for x in forbidden.values()])
    cd = np.asarray([x["collision_signed_distance_m"] for x in contacts])
    margin = np.minimum(v[:14]-info["joint_limits"][:, 0],
                        info["joint_limits"][:, 1]-v[:14])
    screen = float(np.degrees(np.arccos(np.clip(abs(np.dot(prot[:, 0], [1, 0, 0])), -1, 1))))
    long = float(np.degrees(np.arccos(np.clip(abs(np.dot(prot[:, 2], [0, 0, 1])), -1, 1))))
    stability = stability_proxy(points, applied)
    valid = bool(np.all((cd >= CONTACT_RANGE[0]) & (cd <= CONTACT_RANGE[1]))
                 and max(errors) < NORMAL_LIMIT_DEG and inside
                 and fd.min(initial=1) >= -FORBIDDEN_TOL
                 and margin.min() >= ARM_MARGIN and screen < 5 and long < 5)
    return {
        "valid": valid, "contacts": contacts, "forbidden": forbidden,
        "contact_distances": cd, "maximum_contact_normal_error_deg": max(errors),
        "all_contacts_inside": inside,
        "minimum_forbidden_clearance_m": float(fd.min(initial=1)),
        "joint_limit_margins": margin, "minimum_arm_wrist_margin_rad": float(margin.min()),
        "screen_frontal_angle_deg": screen, "long_up_angle_deg": long,
        "stability": stability,
    }


def search_type(kind, info, layout, old_x, old_center, count, max_nfev):
    specs = topology(kind)
    model, _ = old.expanded_phone_model(old_center, np.zeros(3))
    data = mujoco.MjData(model)
    phone_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phone_proxy")
    groups = make_groups(model, specs)
    hr = np.r_[layout["hands"]["left"]["ranges"], layout["hands"]["right"]["ranges"]]
    lo = np.r_[info["joint_limits"][:, 0]+ARM_MARGIN, hr[:, 0]+.005,
               old_center+[-.08, -.035, -.10], np.radians([-4, -4, -4])]
    hi = np.r_[info["joint_limits"][:, 1]-ARM_MARGIN, hr[:, 1]-.005,
               old_center+[.04, .035, .08], np.radians([4, 4, 4])]
    natural = old.relative.load_natural_start(old.NATURAL_NPZ, info)["arm_q"]
    base = np.r_[old_x, old_center, np.zeros(3)]
    def residual(v):
        assign(model, data, info, layout, phone_bid, v)
        prot = Rotation.from_euler("xyz", v[31:34]).as_matrix()
        rr = []
        for spec in specs:
            gid = groups["intended"][(spec["side"], spec["part"])]
            d, _ = refined.distance(model, data, gid, groups["phone"])
            pad = refined.pad_from_mesh(model, data, gid, v[28:31]-data.geom_xpos[gid])
            local = prot.T @ (pad["center"]-v[28:31])
            target = np.asarray(spec["point"])
            rr.append(650*(d-.0001))
            rr.extend(150*(local-target))
        for pairs, weight in ((groups["phone_forbidden"], 260),
                              (groups["robot_pairs"], 200)):
            for a, b in pairs:
                d, _ = refined.distance(model, data, a, b)
                rr.append(weight*min(0, d-FORBIDDEN_TOL))
        rr.extend(.045*(v[:14]-natural))
        rr.extend(.01*(v[14:28]-old_x[14:28]))
        mirror = np.array([1, -1, -1, 1, -1, 1, -1])
        rr.extend(.06*(v[7:14]-mirror*v[:7]))
        rr.extend(2*(v[28:31]-old_center))
        rr.extend(2*v[31:34])
        return np.asarray(rr)
    rows = []
    families = ("natural", "wrist_roll", "wrist_yaw", "elbow_up", "elbow_down",
                "middle", "thumb", "index", "mirror", "phone_pose", "combined")
    for seed in range(count):
        rng = np.random.default_rng(TYPE_SEEDS[kind] + seed)
        v0 = base.copy()
        family = families[seed % len(families)]
        v0[:28] += rng.normal(0, .05+.01*(seed//len(families)), 28)
        if family == "wrist_roll": v0[[4, 11]] += [.3, -.3]
        if family == "wrist_yaw": v0[[6, 13]] += [.25, -.25]
        if family == "elbow_up": v0[[3, 10]] += .35
        if family == "elbow_down": v0[[3, 10]] -= .35
        if family == "middle": v0[[17, 18, 24, 25]] += [-.3, -.3, .3, .3]
        if family == "thumb": v0[[14, 15, 16, 21, 22, 23]] += rng.normal(0, .25, 6)
        if family == "index": v0[[19, 20, 26, 27]] += rng.normal(0, .25, 4)
        if family == "mirror":
            delta = rng.normal(0, .18, 7)
            v0[:7] += delta
            v0[7:14] += np.array([1,-1,-1,1,-1,1,-1])*delta
        v0[28:31] += rng.normal(0, [.015, .008, .025])
        v0[31:34] += rng.normal(0, np.radians(1.5), 3)
        v0 = np.clip(v0, lo, hi)
        sol = least_squares(residual, v0, bounds=(lo, hi), max_nfev=max_nfev,
                            ftol=2e-9, xtol=2e-9, gtol=2e-9)
        rec = evaluate(model, data, info, layout, phone_bid, groups, specs, sol.x)
        rec.update(grasp_type=kind, candidate=seed, family=family,
                   optimizer_cost=float(sol.cost), optimizer_success=bool(sol.success),
                   v=sol.x.copy(), model=model, groups=groups, specs=specs)
        rows.append(rec)
        print(f"{kind} {seed+1:03d}/{count} valid={rec['valid']} "
              f"normal={rec['maximum_contact_normal_error_deg']:.1f} "
              f"forbid={rec['minimum_forbidden_clearance_m']:.5f} "
              f"stab={rec['stability']['score']:.3f}", flush=True)
    return rows


def csv_output(path, rows):
    fields = ("grasp_type candidate family valid optimizer_cost "
              "maximum_contact_normal_error_deg all_contacts_inside "
              "minimum_forbidden_clearance_m minimum_arm_wrist_margin_rad "
              "screen_frontal_angle_deg long_up_angle_deg stability_score "
              "wrench_rank gravity_support").split()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for r in rows:
            w.writerow({
                "grasp_type": r["grasp_type"], "candidate": r["candidate"],
                "family": r["family"], "valid": r["valid"],
                "optimizer_cost": r["optimizer_cost"],
                "maximum_contact_normal_error_deg": r["maximum_contact_normal_error_deg"],
                "all_contacts_inside": r["all_contacts_inside"],
                "minimum_forbidden_clearance_m": r["minimum_forbidden_clearance_m"],
                "minimum_arm_wrist_margin_rad": r["minimum_arm_wrist_margin_rad"],
                "screen_frontal_angle_deg": r["screen_frontal_angle_deg"],
                "long_up_angle_deg": r["long_up_angle_deg"],
                "stability_score": r["stability"]["score"],
                "wrench_rank": r["stability"]["wrench_matrix_rank"],
                "gravity_support": r["stability"]["gravity_support_component"],
            })


def strip(r):
    return {k: v for k, v in r.items()
            if k not in ("v", "model", "groups", "specs", "contacts", "forbidden")}


def save_npz(path, selected, info, layout):
    v, model = selected["v"], selected["model"]
    qpos = model.key_qpos[0].copy()
    qpos[info["arm_qpos_ids"]] = v[:14]
    qpos[layout["hands"]["left"]["qadr"]] = v[14:21]
    qpos[layout["hands"]["right"]["qadr"]] = v[21:28]
    contacts = selected["contacts"]
    fnames = list(selected["forbidden"])
    payload = dict(
        selected_grasp_type=np.asarray(selected["grasp_type"]),
        full_g1_qpos=qpos, arm_qpos=v[:14], left_dex3_qpos=v[14:21],
        right_dex3_qpos=v[21:28],
        phone_proxy_pose=np.r_[v[28:31],
            Rotation.from_euler("xyz", v[31:34]).as_quat()[[3,0,1,2]]],
        intended_contact_points=np.asarray([c["contact_point"] for c in contacts]),
        contact_normals=np.asarray([c["phone_to_fingertip_normal"] for c in contacts]),
        contact_normal_errors=np.asarray([c["normal_error_deg"] for c in contacts]),
        support_contact_points=np.asarray([c["contact_point"] for c in contacts
                                           if "support" in c["role"]]),
        phone_clearances=np.asarray([c["collision_signed_distance_m"] for c in contacts]),
        robot_collision_clearances=np.asarray([
            selected["forbidden"][k]["signed_distance_m"] for k in fnames]),
        forbidden_pair_names=np.asarray(fnames),
        joint_limit_margins=selected["joint_limit_margins"],
        stability_proxy_score=np.asarray(selected["stability"]["score"]))
    tmp = path.with_suffix(".npz.incomplete")
    with tmp.open("wb") as f: np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def main():
    a = arguments()
    if a.candidates_per_type < 100:
        raise ValueError("--candidates-per-type must be >=100")
    a.output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(SOURCE_REPORT.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    old_report = json.loads(old.OLD_REPORT.read_text()) if hasattr(old, "OLD_REPORT") else json.loads(
        (ROOT/"converted_runs/g1_dex3_static_phone_grasp/static_grasp_report.json").read_text())
    best = old_report["best_candidate"]
    old_x, center = np.asarray(best["x"], float), np.asarray(best["phone_center"], float)
    info = old.relative.latest.ik.validate_model(old.G1_XML)
    layout, _ = old.hand_layout(info)
    rows = []
    for kind in TYPES:
        rows += search_type(kind, info, layout, old_x, center,
                            a.candidates_per_type, a.max_nfev)
    csv_output(a.output/"grasp_primitive_candidates.csv", rows)
    valid = [r for r in rows if r["valid"]]
    comparison = {
        "pure_thumb_index_antipodal_pinch_excluded": True,
        "exclusion_reason": prior["largest_blocker"],
        "contact_geometry_source": str(CALIBRATION),
        "force_closure_method": "CONTACT_STABILITY_PROXY only; no physics claim",
        "counts": {k: {"evaluated": sum(r["grasp_type"] == k for r in rows),
                       "hard_constraint_passed": sum(r["grasp_type"] == k and r["valid"] for r in rows)}
                   for k in TYPES},
        "search_configuration": {
            "candidates_per_type": a.candidates_per_type,
            "bounded_local_function_evaluations_per_seed": a.max_nfev,
            "deterministic_seed_bases": TYPE_SEEDS,
            "seed_families": [
                "natural", "wrist_roll", "wrist_yaw", "elbow_up", "elbow_down",
                "middle", "thumb", "index", "mirror", "phone_pose", "combined"],
        },
        "best_by_type": {
            k: strip(min([r for r in rows if r["grasp_type"] == k],
                         key=lambda r: (not r["valid"],
                            r["maximum_contact_normal_error_deg"],
                            -r["stability"]["score"],
                            -r["minimum_forbidden_clearance_m"])))
            for k in TYPES},
    }
    old.atomic_json(a.output/"grasp_primitive_comparison.json", comparison)
    if not valid:
        report = {
            "verdict": "G1_PHONE_GRASP_PRIMITIVES_BLOCKED", "safety_pass": False,
            "candidate_count": len(rows), "comparison": comparison,
            "largest_blocker": (
                "No actual-mesh three-point, bottom-support, or bimanual-side "
                "topology simultaneously passed designated-face normal/inside "
                "contact and forbidden collision constraints."),
            "selected_npz_generated": False, "images_generated": False,
            "trajectory_generated": False, "workspace_calibration": False,
            "isaac_lab_executed": False, "hardware_executed": False,
        }
        old.atomic_json(a.output/"dex3_selected_grasp_primitive.json",
                        {"selected": None, "verdict": report["verdict"]})
        old.atomic_json(a.output/"static_phone_grasp_report.json", report)
        print(json.dumps(old.serial(report), indent=2))
        print("G1_PHONE_GRASP_PRIMITIVES_BLOCKED")
        return 2
    selected = max(valid, key=lambda r: (
        r["stability"]["score"], -r["maximum_contact_normal_error_deg"],
        r["minimum_forbidden_clearance_m"], r["minimum_arm_wrist_margin_rad"]))
    save_npz(a.output/"selected_static_phone_grasp.npz", selected, info, layout)
    chosen = {
        "selected_grasp_type": selected["grasp_type"],
        "candidate": selected["candidate"], "family": selected["family"],
        "contacts": selected["contacts"], "stability": selected["stability"],
        "third_finger_role": [c["role"] for c in selected["contacts"]
                              if c["part"] == "middle"],
    }
    old.atomic_json(a.output/"dex3_selected_grasp_primitive.json", chosen)
    report = {
        "verdict": "G1_PHONE_GRASP_PRIMITIVE_READY", "safety_pass": True,
        "selected": chosen, "minimum_joint_margin_rad": selected["minimum_arm_wrist_margin_rad"],
        "minimum_forbidden_clearance_m": selected["minimum_forbidden_clearance_m"],
        "screen_frontal_angle_deg": selected["screen_frontal_angle_deg"],
        "long_up_angle_deg": selected["long_up_angle_deg"],
        "pure_thumb_index_failure": prior["largest_blocker"],
        "trajectory_generated": False, "workspace_calibration": False,
        "isaac_lab_executed": False, "hardware_executed": False,
        "gui_command": (
            f"{sys.executable} {ROOT/'tools/view_g1_selected_phone_grasp_primitive.py'} "
            f"--grasp {a.output/'selected_static_phone_grasp.npz'}"),
    }
    old.atomic_json(a.output/"static_phone_grasp_report.json", report)
    # Images are generated by the viewer module in offscreen mode.
    import view_g1_selected_phone_grasp_primitive as viewer
    viewer.render_all(a.output/"selected_static_phone_grasp.npz", a.output)
    print(json.dumps(old.serial(report), indent=2))
    print("G1_PHONE_GRASP_PRIMITIVE_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
