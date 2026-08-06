#!/usr/bin/env python3
"""Search the mandatory single-left-hand Dex3 broad-face phone grasp.

The prior bimanual-side-support result is diagnostic-only and is never loaded.
Kinematics only: qpos + mj_forward, no physics or base motion.
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
sys.path[:0] = [str(ROOT), str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old  # noqa: E402
import refine_g1_dex3_static_phone_contact as contact  # noqa: E402
import retarget_episode49_relative_bimanual_neutral_pinch_to_g1 as neutral  # noqa: E402

OUT = ROOT/"converted_runs/smolvla_20k_episode49_role_aware_g1"
EVAL = ROOT/"evaluation/smolvla_episode49_role_aware_g1"
SOURCE = ROOT/("evaluation/smolvla_episode49_temporal_consensus/"
               "episode_000049_temporal_consensus.npz")
PHONE_DIMS = np.array([.1496, .00795, .0715])
CONTACT_RANGE = (-.0002, .001)
FORBIDDEN_TOL = .0002
NORMAL_LIMIT = 55.
MARGIN = .03


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--evaluation", type=Path, default=EVAL)
    p.add_argument("--candidates", type=int, default=160)
    p.add_argument("--max-nfev", type=int, default=40)
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def body(part):
    return f"left_hand_{part}_{2 if part == 'thumb' else 1}_link"


def semantic_evidence() -> dict:
    with np.load(SOURCE, allow_pickle=False) as z:
        raw = z["optimized_action"].astype(float)
        fps = float(z["fps"])
    amodel, _ = old.relative.latest.aloha.load_validated_model(old.relative.latest.ALOHA_XML)
    aq, clipped = old.relative.latest.aloha.mapped_qpos(raw)
    fk = old.relative.latest.aloha.fk(amodel, aq)
    result = {"optimized_action_file": str(SOURCE), "key": "optimized_action",
              "shape": list(raw.shape), "fps": fps, "mapping_clipped_frames": clipped,
              "hard_role_assignment": {
                  "left": "phone_grasp_move_place",
                  "right": "accessory_grasp_remove"},
              "hands": {}}
    for side, col in (("left", 6), ("right", 13)):
        phase, pi = neutral.hysteretic_phase(raw[:, col])
        pos = fk[f"{side}_position_m"]
        result["hands"][side] = {
            "gripper_width_stats_m": old.stats(raw[:, col]),
            "gripper_hysteresis_transitions": pi["transitions"],
            "fk_start_m": pos[0], "fk_end_m": pos[-1],
            "fk_workspace_min_m": pos.min(axis=0), "fk_workspace_max_m": pos.max(axis=0),
            "path_length_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
            "role_interpretation": (
                "Source-relative motion is retained per arm; object identity is "
                "the user-supplied hard semantic assignment, not inferred from geometry.")
        }
    return result


def groups(model):
    phone = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "phone_proxy_geom")
    intended = {p: contact.collision_geoms(model, body(p))[-1]
                for p in ("thumb", "index")}
    collision = [g for g in range(model.ngeom)
                 if model.geom_contype[g] or model.geom_conaffinity[g]]
    forbidden_phone, robot = [], []
    for g in collision:
        if g == phone or g in intended.values():
            continue
        b = old.body_name(model, g)
        if old.category(b) in ("finger", "hand_wrist", "arm"):
            forbidden_phone.append((g, phone))
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
                robot.append((a, b))
    return dict(phone=phone, intended=intended, forbidden_phone=forbidden_phone,
                robot=robot)


def assign(model, data, info, layout, phone_bid, v, parked_right):
    data.qpos[:] = model.key_qpos[0]
    data.qpos[info["arm_qpos_ids"][:7]] = v[:7]
    data.qpos[info["arm_qpos_ids"][7:]] = parked_right["arm"]
    data.qpos[layout["hands"]["left"]["qadr"]] = v[7:14]
    data.qpos[layout["hands"]["right"]["qadr"]] = parked_right["hand"]
    model.body_pos[phone_bid] = v[14:17]
    q = Rotation.from_euler("xyz", v[17:20]).as_quat()
    model.body_quat[phone_bid] = q[[3,0,1,2]]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def narrow_point(data, gid, phone, ft):
    for c in data.contact:
        if {int(c.geom1), int(c.geom2)} == {int(gid), int(phone)}:
            return np.asarray(c.pos).copy()
    return .5*(ft[:3]+ft[3:])


def evaluate(model, data, info, layout, phone_bid, g, v, parked):
    assign(model, data, info, layout, phone_bid, v, parked)
    prot = Rotation.from_euler("xyz", v[17:20]).as_matrix()
    recs = {}
    for part, face in (("thumb", 1), ("index", -1)):
        gid = g["intended"][part]
        dist, ft = contact.distance(model, data, gid, g["phone"])
        normal = contact.contact_normal_phone_to_tip(model, data, gid, g["phone"], ft)
        target_normal = prot[:, 0]*face
        error = float(np.degrees(np.arccos(np.clip(np.dot(normal, target_normal), -1, 1))))
        point = narrow_point(data, gid, g["phone"], ft)
        local = prot.T@(point-v[14:17])
        inside = (abs(local[1]) <= PHONE_DIMS[2]/2-.001
                  and abs(local[2]) <= PHONE_DIMS[0]/2-.001
                  and abs(local[0]-face*PHONE_DIMS[1]/2) <= .001)
        recs[part] = dict(distance=dist, point=point, normal=normal,
                          normal_error_deg=error, phone_local_point=local,
                          inside_broad_face=inside)
    forbidden = {}
    for label, pairs in (("phone", g["forbidden_phone"]), ("robot", g["robot"])):
        for a, b in pairs:
            d, ft = contact.distance(model, data, a, b)
            forbidden[f"{label}:{old.body_name(model,a)}[{a}]|{old.body_name(model,b)}[{b}]"] = {
                "signed_distance_m": d, "closest_points": ft.reshape(2,3)}
    fd = np.array([x["signed_distance_m"] for x in forbidden.values()])
    cd = np.array([recs[p]["distance"] for p in ("thumb", "index")])
    margin = np.minimum(v[:7]-info["joint_limits"][:7,0],
                        info["joint_limits"][:7,1]-v[:7])
    normals_oppose = float(np.dot(recs["thumb"]["normal"], recs["index"]["normal"]))
    gap = float(np.linalg.norm(recs["thumb"]["point"]-recs["index"]["point"]))
    screen = float(np.degrees(np.arccos(np.clip(abs(np.dot(prot[:,0],[1,0,0])),-1,1))))
    long = float(np.degrees(np.arccos(np.clip(abs(np.dot(prot[:,2],[0,0,1])),-1,1))))
    # Wrist curl diagnostic: angle between forearm->wrist and palm/contact midpoint.
    wrist = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_yaw_link")
    fore = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_roll_link")
    a = data.xpos[wrist]-data.xpos[fore]
    b = .5*(recs["thumb"]["point"]+recs["index"]["point"])-data.xpos[wrist]
    wrist_bend = float(np.degrees(np.arccos(np.clip(
        np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12),-1,1))))
    valid = bool(np.all((cd>=CONTACT_RANGE[0])&(cd<=CONTACT_RANGE[1]))
        and max(x["normal_error_deg"] for x in recs.values()) < NORMAL_LIMIT
        and all(x["inside_broad_face"] for x in recs.values())
        and normals_oppose < -.55
        and gap >= PHONE_DIMS[1]-.001 and gap <= PHONE_DIMS[1]+.004
        and fd.min(initial=1) >= -FORBIDDEN_TOL
        and margin.min() >= MARGIN and screen<5 and long<5 and wrist_bend<75)
    return dict(valid=valid, contacts=recs, forbidden=forbidden,
                contact_distances=cd, normal_opposition_dot=normals_oppose,
                maximum_contact_normal_error_deg=max(
                    x["normal_error_deg"] for x in recs.values()),
                all_contacts_inside_broad_faces=all(
                    x["inside_broad_face"] for x in recs.values()),
                contact_range_violation_m=float(np.sum(
                    np.maximum(CONTACT_RANGE[0]-cd, 0)
                    + np.maximum(cd-CONTACT_RANGE[1], 0))),
                thumb_index_gap_m=gap, minimum_forbidden_clearance_m=float(fd.min(initial=1)),
                arm_margins=margin, minimum_left_arm_wrist_margin_rad=float(margin.min()),
                screen_error_deg=screen, long_axis_error_deg=long,
                wrist_bend_deg=wrist_bend)


def main():
    a = arguments()
    a.output.mkdir(parents=True,exist_ok=True);a.evaluation.mkdir(parents=True,exist_ok=True)
    evidence = semantic_evidence()
    info = old.relative.latest.ik.validate_model(old.G1_XML)
    natural = old.relative.load_natural_start(old.NATURAL_NPZ, info)
    layout, _ = old.hand_layout(info)
    # No bimanual grasp artifact is opened anywhere in this program.
    center = np.array([.34,.10,1.02])
    model, _ = old.expanded_phone_model(center,np.zeros(3));data=mujoco.MjData(model)
    phone_bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"phone_proxy")
    g=groups(model)
    parked={"arm":natural["arm_q"][7:].copy(),
            "hand":natural["full_qpos"][layout["hands"]["right"]["qadr"]].copy()}
    ranges=layout["hands"]["left"]["ranges"]
    lo=np.r_[info["joint_limits"][:7,0]+MARGIN,ranges[:,0]+.005,
             [.27,.045,.90],np.radians([-4,-4,-4])]
    hi=np.r_[info["joint_limits"][:7,1]-MARGIN,ranges[:,1]-.005,
             [.42,.18,1.12],np.radians([4,4,4])]
    # XML-derived hand order: thumb0/1/2,middle0/1,index0/1.
    pinch=np.array([-.45,.75,1.05,-1.30,-1.50,-1.05,-.82])
    pinch=np.clip(pinch,ranges[:,0]+.01,ranges[:,1]-.01)
    base=np.r_[natural["arm_q"][:7],pinch,center,np.zeros(3)]
    def residual(v):
        assign(model,data,info,layout,phone_bid,v,parked)
        prot=Rotation.from_euler("xyz",v[17:20]).as_matrix()
        pads={}
        rr=[]
        for part,face in (("thumb",1),("index",-1)):
            gid=g["intended"][part]
            d,_=contact.distance(model,data,gid,g["phone"])
            pad=contact.pad_from_mesh(model,data,gid,v[14:17]-data.geom_xpos[gid])
            local=prot.T@(pad["center"]-v[14:17]);pads[part]=local
            rr.append(750*(d-.0001))
            rr.append(220*(local[0]-face*PHONE_DIMS[1]/2))
            rr.append(180*(local[1]))
            rr.append(140*(local[2]))
        delta=pads["thumb"]-pads["index"]
        rr.extend([260*delta[1],260*delta[2],220*(abs(delta[0])-PHONE_DIMS[1])])
        for pairs,w in ((g["forbidden_phone"],280),(g["robot"],220)):
            for x,y in pairs:
                d,_=contact.distance(model,data,x,y)
                rr.append(w*min(0,d-FORBIDDEN_TOL))
        rr.extend(.05*(v[:7]-natural["arm_q"][:7]))
        rr.extend(.012*(v[7:14]-pinch))
        rr.extend(2*(v[14:17]-center));rr.extend(2*v[17:20])
        return np.asarray(rr)
    families=("natural","wrist_roll","wrist_yaw","elbow_up","elbow_down",
              "thumb","index","middle_fold","phone_pose","combined")
    rows=[]
    for seed in range(a.candidates):
        rng=np.random.default_rng(31001+seed);v0=base.copy();family=families[seed%len(families)]
        v0[:14]+=rng.normal(0,.06+.01*(seed//len(families)),14)
        if family=="wrist_roll":v0[4]+=.35
        if family=="wrist_yaw":v0[6]+=.30
        if family=="elbow_up":v0[3]+=.4
        if family=="elbow_down":v0[3]-=.4
        if family=="thumb":v0[7:10]+=rng.normal(0,.3,3)
        if family=="index":v0[12:14]+=rng.normal(0,.3,2)
        if family=="middle_fold":v0[10:12]+=[-.25,-.25]
        v0[14:17]+=rng.normal(0,[.02,.015,.03]);v0[17:20]+=rng.normal(0,np.radians(1.5),3)
        v0=np.clip(v0,lo,hi)
        sol=least_squares(residual,v0,bounds=(lo,hi),max_nfev=a.max_nfev,
                          ftol=2e-9,xtol=2e-9,gtol=2e-9)
        rec=evaluate(model,data,info,layout,phone_bid,g,sol.x,parked)
        rec.update(candidate=seed,family=family,cost=float(sol.cost),
                   optimizer_success=bool(sol.success),v=sol.x.copy())
        rows.append(rec)
        print(f"left-phone {seed+1:03d}/{a.candidates} valid={rec['valid']} "
              f"normal={max(x['normal_error_deg'] for x in rec['contacts'].values()):.1f} "
              f"gap={rec['thumb_index_gap_m']:.4f} forbid={rec['minimum_forbidden_clearance_m']:.5f}",
              flush=True)
    fields=["candidate","family","valid","cost","optimizer_success","thumb_index_gap_m",
            "normal_opposition_dot","minimum_forbidden_clearance_m",
            "minimum_left_arm_wrist_margin_rad","screen_error_deg","long_axis_error_deg",
            "wrist_bend_deg","maximum_contact_normal_error_deg",
            "all_contacts_inside_broad_faces","contact_range_violation_m"]
    with (a.output/"left_phone_grasp_candidates.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in fields})
    valid=[r for r in rows if r["valid"]]
    report={"semantic_role_evidence":evidence,
        "left_role":"phone_grasp_move_place","right_role":"accessory_grasp_remove",
        "bimanual_side_support_status":(
            "PRESERVED_AS_DIAGNOSTIC_ONLY; forbidden as seed, target, converter input, or trajectory input"),
        "bimanual_side_support_artifact_read":False,
        "pure_left_grasp_family":"thumb-index C-gap broad-face only; middle folded non-contact",
        "candidate_count":len(rows),"hard_constraint_pass_count":len(valid),
        "right_arm_during_search":"validated natural parked pose",
        "phone_proxy":"reachable torso-local temporary proxy; no Isaac world pose",
        "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False}
    if not valid:
        best=min(rows,key=lambda r:(
            r["contact_range_violation_m"],
            max(0,-FORBIDDEN_TOL-r["minimum_forbidden_clearance_m"]),
            not r["all_contacts_inside_broad_faces"],
            r["maximum_contact_normal_error_deg"], r["cost"]))
        report.update(verdict="LEFT_HAND_PHONE_GRASP_NOT_FEASIBLE",safety_pass=False,
            largest_blocker=(
                "No single-left-hand thumb/index C-gap candidate achieved two "
                "opposed in-broad-face contacts with the 7.95 mm phone while "
                "keeping middle/palm/right-hand contacts forbidden."),
            best_candidate={k:v for k,v in best.items()
                            if k not in ("v","contacts","forbidden")})
        # Diagnostic-only reproducibility state. This is explicitly not a
        # selected grasp and must never be consumed by a converter.
        bv=best["v"];bq=model.key_qpos[0].copy()
        bq[info["arm_qpos_ids"][:7]]=bv[:7]
        bq[info["arm_qpos_ids"][7:]]=parked["arm"]
        bq[layout["hands"]["left"]["qadr"]]=bv[7:14]
        bq[layout["hands"]["right"]["qadr"]]=parked["hand"]
        with (a.output/"best_failed_internal_FAILED_DIAGNOSTIC.npz").open("wb") as f:
            np.savez_compressed(f, full_qpos=bq, optimizer_variables=bv,
                left_arm_qpos=bv[:7],left_dex3_qpos=bv[7:14],
                phone_xyz_rpy=bv[14:20],candidate=np.asarray(best["candidate"]),
                family=np.asarray(best["family"]),
                intended_contact_distances=best["contact_distances"])
        old.atomic_json(a.output/"role_aware_grasp_report.json",report)
        print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 2
    selected=min(valid,key=lambda r:(max(x["normal_error_deg"] for x in r["contacts"].values()),
                                    -r["minimum_forbidden_clearance_m"],r["cost"]))
    v=selected["v"];qpos=model.key_qpos[0].copy()
    qpos[info["arm_qpos_ids"][:7]]=v[:7];qpos[info["arm_qpos_ids"][7:]]=parked["arm"]
    qpos[layout["hands"]["left"]["qadr"]]=v[7:14];qpos[layout["hands"]["right"]["qadr"]]=parked["hand"]
    payload=dict(full_qpos=qpos,left_arm_qpos=v[:7],left_dex3_qpos=v[7:14],
        right_parked_arm_qpos=parked["arm"],right_parked_dex3_qpos=parked["hand"],
        phone_proxy_pose=np.r_[v[14:17],Rotation.from_euler("xyz",v[17:20]).as_quat()[[3,0,1,2]]],
        contact_names=np.asarray(["left_thumb","left_index"]),
        contact_points=np.asarray([selected["contacts"][p]["point"] for p in ("thumb","index")]),
        contact_normals=np.asarray([selected["contacts"][p]["normal"] for p in ("thumb","index")]),
        contact_distances=selected["contact_distances"],joint_limit_margins=selected["arm_margins"],
        forbidden_pair_names=np.asarray(list(selected["forbidden"])),
        forbidden_clearances=np.asarray([x["signed_distance_m"] for x in selected["forbidden"].values()]))
    tmp=a.output/"selected_left_phone_grasp.npz.incomplete"
    with tmp.open("wb") as f:np.savez_compressed(f,**payload)
    os.replace(tmp,a.output/"selected_left_phone_grasp.npz")
    report.update(verdict="LEFT_HAND_PHONE_GRASP_FEASIBLE",safety_pass=True,
                  selected_candidate=selected["candidate"],contacts=selected["contacts"],
                  minimum_joint_margin=selected["minimum_left_arm_wrist_margin_rad"],
                  minimum_forbidden_clearance=selected["minimum_forbidden_clearance_m"])
    old.atomic_json(a.output/"role_aware_grasp_report.json",report)
    print(json.dumps(old.serial(report),indent=2));return 0


if __name__=="__main__":
    raise SystemExit(main())
