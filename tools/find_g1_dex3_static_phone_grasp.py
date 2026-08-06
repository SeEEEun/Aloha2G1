#!/usr/bin/env python3
"""Audit G1/Dex3 and search a collision-checked static phone-proxy pinch.

This program is deliberately kinematic: it assigns qpos and calls mj_forward.
It never calls mj_step, never writes an Isaac asset, and never moves the base.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import retarget_episode49_consensus_relative_bimanual_to_g1 as relative  # noqa: E402
import retarget_episode49_relative_bimanual_neutral_pinch_to_g1 as neutral  # noqa: E402

G1_XML = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
RELATIVE_NPZ = ROOT / ("converted_runs/smolvla_20k_episode49_consensus_relative_g1/"
                       "g1_episode49_consensus_relative_trajectory.npz")
NATURAL_NPZ = ROOT / ("converted_runs/magsafe_20260723_162750/dynamic_bimanual_spacing/"
                      "g1_dynamic_bimanual_full_trajectory.npz")
SCENE_LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
EVAL = ROOT / "evaluation/g1_prephysics_motion_validation"
OUT = ROOT / "converted_runs/g1_dex3_static_phone_grasp"
PHONE = np.array([0.1496, 0.00795, 0.0715])  # long, thickness, short
ARM_MARGIN = 0.05
MIN_ARM_MARGIN = 0.03
CONTACT_TOL = 0.001
CLEARANCE = 0.0005
TIP_BODIES = {"thumb": "hand_thumb_2_link", "index": "hand_index_1_link",
              "middle": "hand_middle_1_link"}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--evaluation", type=Path, default=EVAL)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--seeds", type=int, default=18)
    p.add_argument("--max-nfev", type=int, default=1800)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    return x


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(serial(obj), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def body_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or "world"


def category(body: str) -> str:
    if "hand_thumb" in body or "hand_index" in body or "hand_middle" in body:
        return "finger"
    if "wrist" in body or body in ("left_hand", "right_hand"):
        return "hand_wrist"
    if "elbow" in body or "shoulder" in body or "forearm" in body:
        return "arm"
    if body == "torso_link" or "waist" in body or "pelvis" in body:
        return "torso"
    return "other"


def contact_audit(model: mujoco.MjModel, qpos: np.ndarray) -> dict:
    data = mujoco.MjData(model)
    flags = {k: np.zeros(len(qpos), bool) for k in (
        "arm_torso", "hand_hand", "finger_torso", "finger_finger", "other_robot")}
    pairs: dict[str, list[int]] = {}
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for c in data.contact:
            a, b = body_name(model, c.geom1), body_name(model, c.geom2)
            ca, cb = category(a), category(b)
            pair = "|".join(sorted((a, b)))
            # Ignore the model's intended adjacent-link overlap only. These
            # pairs are present in the stand keyframe and are not cross-finger.
            adjacent = (
                a.split("_link")[0] == b.split("_link")[0]
                or ("thumb_1" in a and "thumb_2" in b)
                or ("thumb_2" in a and "thumb_1" in b)
                or ("index_0" in a and "index_1" in b)
                or ("index_1" in a and "index_0" in b)
                or ("middle_0" in a and "middle_1" in b)
                or ("middle_1" in a and "middle_0" in b)
            )
            label = None
            if {ca, cb} == {"arm", "torso"}:
                label = "arm_torso"
            elif ((a.startswith("left_") and b.startswith("right_"))
                  or (a.startswith("right_") and b.startswith("left_"))):
                label = "finger_finger" if ca == cb == "finger" else "hand_hand"
            elif {ca, cb} == {"finger", "torso"}:
                label = "finger_torso"
            elif ca == cb == "finger" and not adjacent:
                label = "finger_finger"
            elif not adjacent and ca != "other" and cb != "other":
                label = "other_robot"
            if label:
                flags[label][t] = True
                pairs.setdefault(pair, []).append(t)
    return {
        "flags": flags,
        "counts": {k: int(v.sum()) for k, v in flags.items()},
        "pairs": {k: {"frame_count": len(set(v)),
                      "first_frame": min(v), "last_frame": max(v)}
                  for k, v in pairs.items()},
    }


def stats(x: np.ndarray) -> dict:
    a = np.asarray(x, float).ravel()
    return {k: float(v) for k, v in (
        ("max", np.max(a, initial=0)), ("p95", np.percentile(a, 95)),
        ("p99", np.percentile(a, 99)), ("mean", np.mean(a)))}


def arm_motion_report(info: dict, path: Path, out: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        qpos = z["full_g1_joint_trajectory"].astype(float)
        arm = z["full_arm"].astype(float)
        task = int(z["task_start_frame"])
        fps = float(z["fps"])
        tlp = z["g1_target_left_position"].astype(float)
        trp = z["g1_target_right_position"].astype(float)
        glp = z["g1_achieved_left_position"].astype(float)
        grp = z["g1_achieved_right_position"].astype(float)
    audit = contact_audit(info["model"], qpos)
    # Arm-only separation: replace Dex3 qpos by the collision-free natural
    # start hand qpos while keeping every arm frame unchanged.
    start = relative.load_natural_start(NATURAL_NPZ, info)
    arm_only = qpos.copy()
    layout, _ = neutral.hand_joint_schema(info)
    for side in ("left", "right"):
        adr = layout["hands"][side]["qadr"]
        arm_only[:, adr] = start["full_qpos"][adr]
    arm_audit = contact_audit(info["model"], arm_only)
    lim = info["joint_limits"]
    margin = np.minimum(arm-lim[:, 0], lim[:, 1]-arm)
    step = np.abs(np.diff(arm, axis=0))
    vel, acc = step*fps, np.abs(np.diff(arm, n=2, axis=0))*fps**2
    tmid, trel = .5*(tlp+trp), trp-tlp
    gmid, grel = .5*(glp+grp), grp-glp
    report = {
        "source": str(path.resolve()), "mode": "qpos + mj_forward only",
        "frame_count": len(qpos), "task_start_frame": task, "fps": fps,
        "hand_distance_m": stats(np.linalg.norm(grel, axis=1)),
        "midpoint_rmse_mm": float(np.sqrt(np.mean(np.sum((gmid-tmid)**2, axis=1))))*1000,
        "relative_vector_rmse_mm": float(np.sqrt(np.mean(np.sum((grel-trel)**2, axis=1))))*1000,
        "joint_limit_violation_count": int(np.sum(margin < -1e-9)),
        "minimum_arm_wrist_joint_margin_rad_full_including_approach": float(margin.min()),
        "minimum_arm_wrist_joint_margin_rad_task": float(margin[task:].min()),
        "nan_inf_count": int(np.size(qpos)-np.isfinite(qpos).sum()),
        "joint_step_rad": stats(step), "joint_velocity_rad_s": stats(vel),
        "joint_acceleration_rad_s2": stats(acc),
        "original_dex3_collision_audit": audit["counts"],
        "original_dex3_collision_pairs": audit["pairs"],
        "arm_only_with_natural_hand_pose_collision_audit": arm_audit["counts"],
        "arm_only_collision_pairs": arm_audit["pairs"],
        "separation_conclusion": (
            "Arm motion is reported independently by holding Dex3 at the "
            "validated natural-start hand qpos; remaining original-only events "
            "are attributable to Dex3 geometry/posture."
        ),
    }
    atomic_json(out, report)
    return report


def hand_layout(info: dict) -> tuple[dict, dict]:
    return neutral.hand_joint_schema(info)


def geom_for_body(model: mujoco.MjModel, body: str, collision_only=True) -> list[int]:
    out = []
    for gid in range(model.ngeom):
        if body_name(model, gid) == body:
            if not collision_only or model.geom_contype[gid] or model.geom_conaffinity[gid]:
                out.append(gid)
    return out


def dex3_calibration(info: dict, start: dict, out: Path) -> tuple[dict, dict]:
    layout, schema = hand_layout(info)
    model, data = layout["model"], mujoco.MjData(layout["model"])
    data.qpos[:] = start["full_qpos"]
    mujoco.mj_forward(model, data)
    result = {
        "g1_xml": str(G1_XML), "natural_start_source": str(NATURAL_NPZ),
        "frame_definition": "computed independently at the natural-start qpos",
        "sides": {},
    }
    frames = {}
    for side in ("left", "right"):
        wrist = f"{side}_wrist_yaw_link"
        palm = wrist
        bids = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{side}_{v}") for k, v in TIP_BODIES.items()}
        wb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist)
        thumb, index = data.xpos[bids["thumb"]].copy(), data.xpos[bids["index"]].copy()
        origin = .5*(thumb+index)
        closing = index-thumb
        closing /= np.linalg.norm(closing)
        wr = data.xmat[wb].reshape(3, 3).copy()
        forward = wr[:, 0].copy()
        forward -= closing*np.dot(forward, closing)
        forward /= np.linalg.norm(forward)
        normal = np.cross(closing, forward)
        normal /= np.linalg.norm(normal)
        # Make left/right normal directions consistent with torso up.
        if np.dot(normal, [0, 0, 1]) < 0:
            normal, forward = -normal, -forward
        frames[side] = {"origin": origin, "closing": closing,
                        "forward": forward, "normal": normal}
        result["sides"][side] = {
            "wrist_link": wrist, "palm_hand_base_link": palm,
            "thumb_links": [f"{side}_hand_thumb_{i}_link" for i in range(3)],
            "index_links": [f"{side}_hand_index_{i}_link" for i in range(2)],
            "middle_third_finger_links": [f"{side}_hand_middle_{i}_link" for i in range(2)],
            "fingertip_links": {k: mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, v) for k, v in bids.items()},
            "joints": {name: {"qpos_index": int(layout["hands"][side]["qadr"][rec["index"]]),
                              "range_rad": rec["range"], "local_axis": rec["axis"],
                              "positive_direction": (
                                  "right-hand rule about XML local joint axis")}
                       for name, rec in schema[side].items()},
            "palm_frame": {"origin_m": data.xpos[wb], "rotation_world": wr,
                           "finger_forward_axis": wr[:, 0],
                           "palm_normal": normal},
            "pinch_frame": frames[side],
        }
    atomic_json(out, result)
    return result, frames


def axes_plot(path: Path, model: mujoco.MjModel, qpos: np.ndarray,
              frames: dict, sides: tuple[str, ...]) -> None:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"closing": "r", "forward": "g", "normal": "b"}
    for side in sides:
        f = frames[side]
        o = f["origin"]
        ax.scatter(*o, label=side)
        for key, color in colors.items():
            v = f[key]
            ax.quiver(*o, *v, length=.08, color=color)
        for suffix in TIP_BODIES.values():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{suffix}")
            ax.scatter(*data.xpos[bid], s=20)
    ax.set(xlabel="torso forward X", ylabel="torso lateral Y", zlabel="torso up Z")
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def expanded_phone_model(phone_center: np.ndarray, phone_rpy: np.ndarray) -> tuple[mujoco.MjModel, Path]:
    base = mujoco.MjModel.from_xml_path(str(G1_XML))
    temp_dir = Path(tempfile.mkdtemp(prefix="g1_phone_proxy_"))
    expanded = temp_dir / "g1_phone_proxy.xml"
    mujoco.mj_saveLastXML(str(expanded), base)
    text = expanded.read_text()
    assets = G1_XML.parent / "assets"
    text = text.replace('meshdir="assets/"', f'meshdir="{assets}/"')
    euler = " ".join(f"{v:.12g}" for v in phone_rpy)
    pos = " ".join(f"{v:.12g}" for v in phone_center)
    # Model axes: X forward/thickness, Y lateral/short, Z up/long.
    geom = (
        f'<body name="phone_proxy" pos="{pos}" euler="{euler}">'
        f'<geom name="phone_proxy_geom" type="box" '
        f'size="{PHONE[1]/2} {PHONE[2]/2} {PHONE[0]/2}" '
        'rgba="0.05 0.25 0.95 0.55" contype="1" conaffinity="1"/>'
        '</body>')
    text = text.replace("<worldbody>", "<worldbody>\n    " + geom, 1)
    expanded.write_text(text)
    return mujoco.MjModel.from_xml_path(str(expanded)), temp_dir


def poses_from_xml(layout: dict, schema: dict) -> dict:
    old = relative.latest.poses(layout["model"], layout["hands"])
    result = {}
    for side in ("left", "right"):
        sign = -1 if side == "left" else 1
        op = old[side][0].copy()
        pre = op.copy()
        pinch = op.copy()
        vals_pre = (-.28, -sign*.55, -sign*.62, sign*.95, sign*1.25,
                    sign*.55, sign*.45)
        vals_pinch = (-.48, -sign*.78, -sign*1.08, sign*1.35, sign*1.55,
                      sign*1.10, sign*.82)
        # qpos order was read from XML: thumb0/1/2,middle0/1,index0/1.
        for arr, vals in ((pre, vals_pre), (pinch, vals_pinch)):
            for i, value in enumerate(vals):
                lo, hi = layout["hands"][side]["ranges"][i]
                arr[i] = np.clip(value, lo+.01, hi-.01)
        result[side] = {"open": op, "pregrasp": pre, "pinch_seed": pinch}
    return result


def solve_candidates(info: dict, start: dict, seeds: int, max_nfev: int) -> tuple[list[dict], dict | None]:
    layout, schema = hand_layout(info)
    primitives = poses_from_xml(layout, schema)
    # Reachable torso-local proxy range, intentionally unrelated to Isaac pose.
    phone_centers = [np.array([x, 0., z]) for x in (.30, .33, .36)
                     for z in (.95, 1.00, 1.05)]
    results = []
    for center_i, center in enumerate(phone_centers):
        model, _ = expanded_phone_model(center, np.zeros(3))
        data = mujoco.MjData(model)
        arm_ids = info["arm_qpos_ids"]
        laddr, raddr = layout["hands"]["left"]["qadr"], layout["hands"]["right"]["qadr"]
        # Expanded model preserves joint/qpos order.
        ranges_l = layout["hands"]["left"]["ranges"]
        ranges_r = layout["hands"]["right"]["ranges"]
        lower = np.r_[info["joint_limits"][:, 0]+ARM_MARGIN,
                      ranges_l[:, 0]+.01, ranges_r[:, 0]+.01]
        upper = np.r_[info["joint_limits"][:, 1]-ARM_MARGIN,
                      ranges_l[:, 1]-.01, ranges_r[:, 1]-.01]
        phone_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     "phone_proxy_geom")
        tip_gids = []
        for side in ("left", "right"):
            tip_gids.extend([
                geom_for_body(model, f"{side}_hand_thumb_2_link")[-1],
                geom_for_body(model, f"{side}_hand_index_1_link")[-1]])
        protected = [g for g in range(model.ngeom)
                     if (model.geom_contype[g] or model.geom_conaffinity[g])
                     and category(body_name(model, g)) in
                     ("finger", "hand_wrist", "arm", "torso")]
        pairs = []
        for i, ga in enumerate(protected):
            for gb in protected[i+1:]:
                a, b = body_name(model, ga), body_name(model, gb)
                if a == b:
                    continue
                cross = (a.startswith("left_") and b.startswith("right_")) or (
                    a.startswith("right_") and b.startswith("left_"))
                torso_pair = "torso_link" in (a, b)
                if cross or torso_pair:
                    pairs.append((ga, gb))
        def assign(x):
            data.qpos[:] = model.key_qpos[0]
            data.qpos[arm_ids] = x[:14]
            data.qpos[laddr] = x[14:21]
            data.qpos[raddr] = x[21:28]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
        def distance(ga, gb):
            return float(mujoco.mj_geomDistance(model, data, ga, gb, .20, None))
        for seed in range(max(1, seeds // len(phone_centers))):
            rng = np.random.default_rng(1000*center_i+seed)
            x0 = np.r_[start["arm_q"], primitives["left"]["pinch_seed"],
                       primitives["right"]["pinch_seed"]]
            # Required seed families: natural, elbow variants, wrist roll,
            # mirrored perturbations, nominal task-ready.
            if seed % 4 == 1:
                x0[[3, 10]] += .35
            elif seed % 4 == 2:
                x0[[4, 11]] += np.array([.45, -.45])
            elif seed % 4 == 3:
                delta = rng.normal(0, .18, 7)
                mirror = np.array([1, -1, -1, 1, -1, 1, -1])
                x0[:7] += delta
                x0[7:14] += mirror*delta
            x0 = np.clip(x0, lower, upper)
            # Tip-center targets merely select the correct contact branch.
            # Exact validity below uses collision meshes and signed distance.
            targets = []
            for side_sign in (1., -1.):
                y = side_sign*(PHONE[2]/2-.004)
                targets += [center+np.array([PHONE[1]/2+.010, y, 0]),
                            center+np.array([-PHONE[1]/2-.010, y, 0])]
            def residual(x):
                assign(x)
                r = []
                for gid, target in zip(tip_gids, targets):
                    r.extend(30*(data.geom_xpos[gid]-target))
                    r.append(80*distance(gid, phone_gid))
                # Keep all non-tip fingers outside the proxy.
                for gid in protected:
                    if gid in tip_gids or body_name(model, gid) == "torso_link":
                        continue
                    r.append(50*min(0., distance(gid, phone_gid)-CLEARANCE))
                for ga, gb in pairs:
                    r.append(35*min(0., distance(ga, gb)-CLEARANCE))
                r.extend(.07*(x[:14]-start["arm_q"]))
                r.extend(.025*(x[14:21]-primitives["left"]["pinch_seed"]))
                r.extend(.025*(x[21:]-primitives["right"]["pinch_seed"]))
                mirror = np.array([1, -1, -1, 1, -1, 1, -1])
                r.extend(.10*(x[7:14]-mirror*x[:7]))
                return np.asarray(r)
            sol = least_squares(residual, x0, bounds=(lower, upper),
                                max_nfev=max_nfev, ftol=2e-10, xtol=2e-10,
                                gtol=2e-10)
            assign(sol.x)
            tip_dist = np.array([distance(g, phone_gid) for g in tip_gids])
            other_phone = [distance(g, phone_gid) for g in protected
                           if g not in tip_gids and body_name(model, g) != "torso_link"]
            cross_clear = [distance(a, b) for a, b in pairs]
            contacts = []
            robot_contacts = []
            penetration = []
            for c in data.contact:
                a, b = body_name(model, c.geom1), body_name(model, c.geom2)
                if "phone_proxy" in (a, b):
                    penetration.append(float(c.dist))
                    if ("thumb_2" in a or "index_1" in a or
                            "thumb_2" in b or "index_1" in b):
                        contacts.append((a, b, float(c.dist)))
                else:
                    robot_contacts.append((a, b, float(c.dist)))
            arm_margin = np.minimum(sol.x[:14]-info["joint_limits"][:, 0],
                                    info["joint_limits"][:, 1]-sol.x[:14])
            # A valid kinematic contact has non-positive/touching signed
            # distance, but no more than CONTACT_TOL mesh interpenetration.
            valid_contact = np.all(np.abs(tip_dist) <= CONTACT_TOL)
            no_penetration = min(penetration, default=0.) >= -CONTACT_TOL
            no_other_phone = min(other_phone, default=1.) >= -CONTACT_TOL
            no_robot = min(cross_clear, default=1.) >= -CONTACT_TOL
            valid = bool(valid_contact and no_penetration and no_other_phone
                         and no_robot and arm_margin.min() >= MIN_ARM_MARGIN)
            rec = {
                "candidate": len(results), "phone_center": center,
                "seed_family": ["natural", "elbow", "wrist_roll", "mirror"][seed % 4],
                "optimizer_success": bool(sol.success), "cost": float(sol.cost),
                "arm_margin": float(arm_margin.min()),
                "tip_phone_signed_distances": tip_dist,
                "minimum_other_phone_clearance": min(other_phone, default=1.),
                "minimum_robot_clearance": min(cross_clear, default=1.),
                "minimum_phone_contact_distance": min(penetration, default=0.),
                "valid": valid, "x": sol.x.copy(), "model": model,
                "primitives": primitives,
            }
            results.append(rec)
            print(f"candidate {rec['candidate']:02d} {rec['seed_family']} "
                  f"center={center} valid={valid} tip={tip_dist} "
                  f"robot={rec['minimum_robot_clearance']:.5f}", flush=True)
    valid = [r for r in results if r["valid"]]
    selected = max(valid, key=lambda r: (
        r["minimum_robot_clearance"], r["arm_margin"],
        -np.max(np.abs(r["tip_phone_signed_distances"])), -r["cost"])) if valid else None
    return results, selected


def write_candidates(path: Path, candidates: list[dict]) -> None:
    fields = ["candidate", "seed_family", "phone_x", "phone_y", "phone_z",
              "valid", "cost", "arm_margin", "max_abs_tip_phone_distance",
              "minimum_other_phone_clearance", "minimum_robot_clearance",
              "minimum_phone_contact_distance"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in candidates:
            w.writerow({
                "candidate": r["candidate"], "seed_family": r["seed_family"],
                "phone_x": r["phone_center"][0], "phone_y": r["phone_center"][1],
                "phone_z": r["phone_center"][2], "valid": r["valid"],
                "cost": r["cost"], "arm_margin": r["arm_margin"],
                "max_abs_tip_phone_distance": np.max(np.abs(r["tip_phone_signed_distances"])),
                "minimum_other_phone_clearance": r["minimum_other_phone_clearance"],
                "minimum_robot_clearance": r["minimum_robot_clearance"],
                "minimum_phone_contact_distance": r["minimum_phone_contact_distance"],
            })


def render_pose(model: mujoco.MjModel, qpos: np.ndarray, path: Path,
                azimuth: float, elevation: float, label: str) -> None:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=800, height=700)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.20, 0., .95]
    cam.distance = 1.25
    cam.azimuth, cam.elevation = azimuth, elevation
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    renderer.close()
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb)
    ImageDraw.Draw(im).text((18, 18), label, fill=(255, 255, 255))
    im.save(path)


def selected_payload(info: dict, start: dict, selected: dict, out: Path) -> dict:
    model, x = selected["model"], selected["x"]
    layout, _ = hand_layout(info)
    qpos = model.key_qpos[0].copy()
    qpos[info["arm_qpos_ids"]] = x[:14]
    qpos[layout["hands"]["left"]["qadr"]] = x[14:21]
    qpos[layout["hands"]["right"]["qadr"]] = x[21:]
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    phone_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "phone_proxy_geom")
    tips, contact_points, contact_normals = {}, [], []
    for side in ("left", "right"):
        tips[side] = {}
        for kind in ("thumb", "index", "middle"):
            body = f"{side}_{TIP_BODIES[kind]}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            gids = geom_for_body(model, body)
            tips[side][kind] = data.xpos[bid].copy()
            if kind != "middle":
                fromto = np.empty(6)
                dist = mujoco.mj_geomDistance(model, data, gids[-1], phone_gid, .2, fromto)
                contact_points.append(.5*(fromto[:3]+fromto[3:]))
                n = fromto[3:]-fromto[:3]
                contact_normals.append(n/(np.linalg.norm(n)+1e-12))
    arm_margin = np.minimum(x[:14]-info["joint_limits"][:, 0],
                            info["joint_limits"][:, 1]-x[:14])
    wrist_pose, palm_pose = {}, {}
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{side}_wrist_yaw_link")
        wrist_pose[side] = np.r_[data.xpos[bid], data.xquat[bid]]
        palm_pose[side] = wrist_pose[side].copy()
    aperture = {s: float(np.linalg.norm(tips[s]["thumb"]-tips[s]["index"]))
                for s in ("left", "right")}
    payload = {
        "g1_root_pose": qpos[:7], "full_g1_qpos": qpos,
        "arm_qpos": x[:14], "left_dex3_qpos": x[14:21],
        "right_dex3_qpos": x[21:], "phone_proxy_pose": np.r_[selected["phone_center"], 1, 0, 0, 0],
        "left_wrist_pose": wrist_pose["left"], "right_wrist_pose": wrist_pose["right"],
        "left_palm_pose": palm_pose["left"], "right_palm_pose": palm_pose["right"],
        "left_thumb_tip_pose": tips["left"]["thumb"],
        "left_index_tip_pose": tips["left"]["index"],
        "right_thumb_tip_pose": tips["right"]["thumb"],
        "right_index_tip_pose": tips["right"]["index"],
        "left_middle_tip_pose": tips["left"]["middle"],
        "right_middle_tip_pose": tips["right"]["middle"],
        "contact_points": np.asarray(contact_points),
        "contact_normals": np.asarray(contact_normals),
        "thumb_index_aperture": np.array([aperture["left"], aperture["right"]]),
        "joint_limit_margins": arm_margin,
        "collision_clearances": np.array([
            selected["minimum_robot_clearance"],
            selected["minimum_other_phone_clearance"],
            selected["minimum_phone_contact_distance"]]),
        "natural_start_qpos": start["full_qpos"],
    }
    tmp = out.with_suffix(".npz.incomplete")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, out)
    return {"qpos": qpos, "tips": tips, "aperture": aperture,
            "arm_margin": arm_margin, "wrist_pose": wrist_pose,
            "palm_pose": palm_pose, "payload": payload}


def main() -> int:
    a = arguments()
    for p in (G1_XML, RELATIVE_NPZ, NATURAL_NPZ, SCENE_LAYOUT):
        if not p.exists():
            raise FileNotFoundError(p)
    a.output.mkdir(parents=True, exist_ok=True)
    a.evaluation.mkdir(parents=True, exist_ok=True)
    info = relative.latest.ik.validate_model(G1_XML)
    start = relative.load_natural_start(NATURAL_NPZ, info)
    motion = arm_motion_report(info, RELATIVE_NPZ,
                               a.evaluation / "arm_motion_report.json")
    calibration, frames = dex3_calibration(
        info, start, a.evaluation / "dex3_kinematic_calibration.json")
    layout, schema = hand_layout(info)
    axes_plot(a.evaluation / "dex3_left_axes.png", layout["model"],
              start["full_qpos"], frames, ("left",))
    axes_plot(a.evaluation / "dex3_right_axes.png", layout["model"],
              start["full_qpos"], frames, ("right",))
    axes_plot(a.evaluation / "dex3_bimanual_axes.png", layout["model"],
              start["full_qpos"], frames, ("left", "right"))
    scene = json.loads(SCENE_LAYOUT.read_text())
    measured = np.asarray(scene["phone"]["size_landscape_xyz"], float)
    if not np.allclose(measured, PHONE, atol=1e-9):
        raise RuntimeError(f"Phone asset dimensions changed: {measured}")
    candidates, selected = solve_candidates(info, start, a.seeds, a.max_nfev)
    write_candidates(a.output / "static_grasp_candidates.csv", candidates)
    primitives = poses_from_xml(layout, schema)
    atomic_json(a.output / "dex3_open_pregrasp_pinch.json", {
        "joint_names": {s: layout["hands"][s]["names"] for s in ("left", "right")},
        "qpos_addresses": {s: layout["hands"][s]["qadr"] for s in ("left", "right")},
        "poses": primitives, "source_scalar_width_m": 0.044,
        "hysteresis": {"open_threshold": neutral.OPEN_THRESHOLD,
                       "pinch_threshold": neutral.PINCH_THRESHOLD,
                       "phase_slew_per_frame": neutral.PHASE_SLEW},
        "note": "PINCH is replaced by selected optimized qpos when a valid candidate exists.",
    })
    proxy = {
        "purpose": "torso-local kinematic calibration pose; not Isaac scene placement",
        "dimensions": {"long_m": measured[0], "thickness_m": measured[1],
                       "short_m": measured[2]},
        "axes": {"long": "+torso_Z", "thickness": "+torso_X",
                 "short": "+torso_Y", "screen_normal": "-torso_X"},
        "candidate_range": {"x_m": [.30, .36], "y_m": [0, 0],
                            "z_m": [.95, 1.05], "rpy_rad": [0, 0, 0]},
        "selected_center_m": selected["phone_center"] if selected else None,
    }
    atomic_json(a.output / "static_phone_proxy_pose.json", proxy)
    if selected is None:
        report = {
            "verdict": "G1_STATIC_PHONE_GRASP_NOT_FOUND", "safety_pass": False,
            "final_prephysics_verdict": "G1_PREPHYSICS_GRASP_SAFETY_BLOCKED",
            "static_grasp_success": False, "trajectory_generation_allowed": False,
            "largest_blocker": (
                "No candidate simultaneously achieved four thumb/index mesh "
                "contacts, zero phone/non-contact penetration, zero cross-hand/"
                "torso collision, and >=0.03 rad arm/wrist margin."),
            "candidate_count": len(candidates),
            "best_candidate": min(candidates, key=lambda r: r["cost"]),
            "phone_dimensions_verified_from": str(SCENE_LAYOUT),
            "arm_motion_report": motion,
            "isaac_lab_executed": False, "base_moved": False,
            "mujoco_mode": "qpos + mj_forward only",
        }
        report["best_candidate"].pop("model", None)
        report["best_candidate"].pop("primitives", None)
        atomic_json(a.output / "static_grasp_report.json", report)
        print(json.dumps(serial(report), indent=2))
        print("G1_STATIC_PHONE_GRASP_NOT_FOUND")
        return 2
    selected_data = selected_payload(
        info, start, selected, a.output / "selected_static_grasp.npz")
    for name, az, el in (("front", 180, 0), ("top", 180, -88),
                         ("side", 90, 0), ("contacts", 165, -12)):
        render_pose(selected["model"], selected_data["qpos"],
                    a.output / f"static_grasp_{name}.png", az, el,
                    f"static Dex3 phone pinch | {name}")
        render_pose(selected["model"], selected_data["qpos"],
                    a.output / f"selected_static_grasp_{name}.png", az, el,
                    f"selected static Dex3 phone pinch | {name}")
    report = {
        "verdict": "G1_STATIC_PHONE_GRASP_FOUND", "safety_pass": True,
        "static_grasp_success": True, "trajectory_generation_allowed": True,
        "selected_candidate": selected["candidate"],
        "phone_proxy_center_torso_local_m": selected["phone_center"],
        "phone_dimensions_m": measured,
        "thumb_index_aperture_m": selected_data["aperture"],
        "third_finger_pose_qpos": {
            "left": selected["x"][17:19], "right": selected["x"][24:26]},
        "minimum_arm_wrist_joint_margin_rad": float(selected_data["arm_margin"].min()),
        "minimum_robot_clearance_m": selected["minimum_robot_clearance"],
        "minimum_other_finger_phone_clearance_m": selected["minimum_other_phone_clearance"],
        "tip_phone_signed_distances_m": selected["tip_phone_signed_distances"],
        "root_motion": False, "isaac_lab_executed": False,
        "mujoco_mode": "qpos + mj_forward only",
        "gui_command": (
            f"{sys.executable} {ROOT/'tools/play_g1_static_phone_grasp_mujoco.py'} "
            f"--grasp {a.output/'selected_static_grasp.npz'}"),
    }
    atomic_json(a.output / "static_grasp_report.json", report)
    print(json.dumps(serial(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
