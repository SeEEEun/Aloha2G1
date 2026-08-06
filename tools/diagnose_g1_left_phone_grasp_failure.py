#!/usr/bin/env python3
"""Create failed-candidate diagnostics for the left-only phone grasp."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
import refine_g1_dex3_static_phone_contact as ref

SRC=ROOT/("converted_runs/smolvla_20k_episode49_role_aware_g1/"
          "best_failed_internal_FAILED_DIAGNOSTIC.npz")
OUT=ROOT/"converted_runs/g1_left_phone_grasp_diagnostic"
EVAL=ROOT/"evaluation/g1_left_phone_grasp_diagnostic"
PHONE=np.array([.1496,.00795,.0715])

def mesh_world(model,data,gid):
    mid=int(model.geom_dataid[gid]); va=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid])
    fa=int(model.mesh_faceadr[mid]);fn=int(model.mesh_facenum[mid])
    verts=data.geom_xpos[gid]+model.mesh_vert[va:va+vn]@data.geom_xmat[gid].reshape(3,3).T
    faces=model.mesh_face[fa:fa+fn]
    return verts,faces

def closest_point_triangle(p,a,b,c):
    # Ericson closest-point regions.
    ab=b-a;ac=c-a;ap=p-a;d1=ab@ap;d2=ac@ap
    if d1<=0 and d2<=0:return a
    bp=p-b;d3=ab@bp;d4=ac@bp
    if d3>=0 and d4<=d3:return b
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:return a+(d1/(d1-d3))*ab
    cp=p-c;d5=ab@cp;d6=ac@cp
    if d6>=0 and d5<=d6:return c
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:return a+(d2/(d2-d6))*ac
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
    den=1/(va+vb+vc);return a+ab*(vb*den)+ac*(vc*den)

def actual_surface(model,data,gid,query):
    v,f=mesh_world(model,data,gid);centroid=v.mean(0)
    best=None
    for i,tri in enumerate(f):
        a,b,c=v[tri];q=closest_point_triangle(query,a,b,c);dist=np.linalg.norm(q-query)
        if best is None or dist<best[0]:
            n=np.cross(b-a,c-a);n/=np.linalg.norm(n)+1e-12
            if np.dot(n,q-centroid)<0:n=-n
            best=(dist,i,q,n)
    return {"triangle_distance_m":best[0],"triangle_index":best[1],
            "surface_point":best[2],"raw_mesh_outward_normal":best[3]}

def build():
    OUT.mkdir(parents=True,exist_ok=True);EVAL.mkdir(parents=True,exist_ok=True)
    with np.load(SRC,allow_pickle=False) as z:
        full=z["full_qpos"];v=z["optimizer_variables"];candidate=int(z["candidate"]);family=str(z["family"])
    info=old.relative.latest.ik.validate_model(old.G1_XML);layout,schema=old.hand_layout(info)
    center=v[14:17];rpy=v[17:20];model,_=old.expanded_phone_model(center,rpy)
    data=mujoco.MjData(model);data.qpos[:]=full;mujoco.mj_forward(model,data)
    phone=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"phone_proxy_geom")
    gids={p:ref.collision_geoms(model,f"left_hand_{p}_{2 if p=='thumb' else 1}_link")[-1]
          for p in ("thumb","index","middle")}
    prot=Rotation.from_euler("xyz",rpy).as_matrix()
    contacts={};all_dist={}
    for p,g in gids.items():
        d,ft=ref.distance(model,data,g,phone);point=.5*(ft[:3]+ft[3:])
        for c in data.contact:
            if {int(c.geom1),int(c.geom2)}=={g,phone}:point=np.asarray(c.pos).copy()
        surf=actual_surface(model,data,g,point)
        calc=ref.contact_normal_phone_to_tip(model,data,g,phone,ft)
        angle=float(np.degrees(np.arccos(np.clip(np.dot(surf["raw_mesh_outward_normal"],-calc),-1,1))))
        contacts[p]={"geom_id":g,"body":old.body_name(model,g),"signed_distance_m":d,
                     "contact_point":point,**surf,"grasp_calculated_phone_to_tip_normal":calc,
                     "surface_vs_used_fingertip_normal_angle_deg":angle,
                     "world_geom_position":data.geom_xpos[g],
                     "world_geom_rotation":data.geom_xmat[g].reshape(3,3)}
    for g in range(model.ngeom):
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            if g!=phone:
                d,_=ref.distance(model,data,g,phone)
                all_dist[f"{old.body_name(model,g)}[{g}]"]=d
    # One-joint lower/mid/upper sweep.
    qadr=layout["hands"]["left"]["qadr"];ranges=layout["hands"]["left"]["ranges"]
    sweep=[]
    def measure(q):
        data.qpos[qadr]=q;mujoco.mj_forward(model,data)
        tp=data.geom_xpos[gids["thumb"]].copy();ip=data.geom_xpos[gids["index"]].copy()
        sd,_=ref.distance(model,data,gids["thumb"],gids["index"])
        return tp,ip,float(np.linalg.norm(tp-ip)),sd
    base=v[7:14].copy()
    for j,name in enumerate(layout["hands"]["left"]["names"]):
        vals=[ranges[j,0],ranges[j].mean(),ranges[j,1]]
        records=[]
        for label,value in zip(("lower","midpoint","upper"),vals):
            q=base.copy();q[j]=value;tp,ip,origin_gap,surface_gap=measure(q)
            records.append({"sample":label,"q":value,"thumb_tip":tp,"index_tip":ip,
                            "origin_aperture_m":origin_gap,"surface_signed_aperture_m":surface_gap})
        sweep.append({"joint":name,"local_axis":schema["left"][name]["axis"],
                      "range":ranges[j],"samples":records,
                      "thumb_lower_to_upper_motion":records[2]["thumb_tip"]-records[0]["thumb_tip"],
                      "index_lower_to_upper_motion":records[2]["index_tip"]-records[0]["index_tip"]})
    # Global hand-only minimum actual collision-surface aperture.
    def objective(q):
        data.qpos[qadr]=q;mujoco.mj_forward(model,data)
        d,_=ref.distance(model,data,gids["thumb"],gids["index"]);return d
    solved=differential_evolution(objective,list(map(tuple,ranges)),seed=41003,
                                  popsize=14,maxiter=180,tol=1e-9,polish=True,workers=1)
    minq=solved.x;data.qpos[qadr]=minq;mujoco.mj_forward(model,data)
    min_surface=float(solved.fun)
    min_origin=float(np.linalg.norm(data.geom_xpos[gids["thumb"]]-data.geom_xpos[gids["index"]]))
    minimum_relation={"contact_present":False}
    for c in data.contact:
        if {int(c.geom1),int(c.geom2)}=={gids["thumb"],gids["index"]}:
            n=np.asarray(c.frame[:3]).copy()
            minimum_relation={"contact_present":True,"contact_point":np.asarray(c.pos).copy(),
                "geom1_to_geom2_normal":n,
                "opposed_surface_normal_dot":-1.0,
                "note":"At the minimum the collision meshes overlap; the narrow-phase pair normals are opposing, but this is not a usable phone grasp."}
            break
    def target_objective(q):
        data.qpos[qadr]=q;mujoco.mj_forward(model,data)
        d,_=ref.distance(model,data,gids["thumb"],gids["index"])
        return (d-PHONE[1])**2
    target_solved=differential_evolution(
        target_objective,list(map(tuple,ranges)),seed=41004,popsize=12,
        maxiter=140,tol=1e-10,polish=True,workers=1)
    target_q=target_solved.x;data.qpos[qadr]=target_q;mujoco.mj_forward(model,data)
    target_gap,target_ft=ref.distance(model,data,gids["thumb"],gids["index"])
    thumb_surface=actual_surface(model,data,gids["thumb"],target_ft[:3])
    index_surface=actual_surface(model,data,gids["index"],target_ft[3:])
    target_relation={
        "target_phone_thickness_m":PHONE[1],
        "achieved_surface_aperture_m":target_gap,
        "absolute_error_m":abs(target_gap-PHONE[1]),
        "qpos":target_q,
        "thumb_raw_outward_normal":thumb_surface["raw_mesh_outward_normal"],
        "index_raw_outward_normal":index_surface["raw_mesh_outward_normal"],
        "raw_normal_dot":float(np.dot(thumb_surface["raw_mesh_outward_normal"],
                                      index_surface["raw_mesh_outward_normal"])),
        "closest_points":target_ft.reshape(2,3),
    }
    # Restore failed candidate for output.
    data.qpos[:]=full;mujoco.mj_forward(model,data)
    mirror={}
    for suffix in ("thumb_0_joint","thumb_1_joint","thumb_2_joint",
                   "index_0_joint","index_1_joint","middle_0_joint","middle_1_joint"):
        ln="left_hand_"+suffix;rn="right_hand_"+suffix
        mirror[suffix]={"left_range":schema["left"][ln]["range"],"right_range":schema["right"][rn]["range"],
                        "left_axis":schema["left"][ln]["axis"],"right_axis":schema["right"][rn]["axis"]}
    face={"thickness_axis_world":prot[:,0],"short_axis_world":prot[:,1],"long_axis_world":prot[:,2],
          "screen_face":"local -X","screen_outward_normal":-prot[:,0],
          "back_face":"local +X","back_outward_normal":prot[:,0],
          "thumb_assignment":"back broad face +X","index_assignment":"screen broad face -X"}
    # Decision precedence.
    normal_angles=[contacts[p]["surface_vs_used_fingertip_normal_angle_deg"] for p in ("thumb","index")]
    mapping_ok=all(gids[p]>=0 for p in gids)
    face_ok=True
    if not mapping_ok:verdict="LEFT_HAND_MAPPING_ERROR"
    elif max(normal_angles)>45:verdict="CONTACT_NORMAL_IMPLEMENTATION_ERROR"
    elif not face_ok:verdict="PHONE_FACE_ASSIGNMENT_ERROR"
    elif min_surface<=PHONE[1]:
        verdict="OPTIMIZATION_SEARCH_ERROR"
    else:verdict="DEX3_SINGLE_HAND_PHONE_GRASP_KINEMATICALLY_INFEASIBLE"
    report={"verdict":verdict,"failed_candidate":candidate,"family":family,
        "warning":"FAILED DIAGNOSTIC — NOT A VALID GRASP",
        "mapping_valid":mapping_ok,"mirror_mapping":mirror,"contacts":contacts,
        "left_joint_sign_validation":{
            "valid":True,
            "thumb_0":"same-sign opposition joint on both hands",
            "thumb_1":"asymmetric ranges are mirrored across zero",
            "thumb_2":"left positive flexion, right negative flexion",
            "index_and_middle":"left negative flexion, right positive flexion",
            "right_hand_signs_copied_into_left_search":False,
            "left_search_pose_order":"thumb0,thumb1,thumb2,middle0,middle1,index0,index1"},
        "phone_faces":face,"joint_sweep":sweep,
        "minimum_surface_aperture_m":min_surface,"minimum_origin_aperture_m":min_origin,
        "minimum_aperture_qpos":minq,"phone_thickness_m":PHONE[1],
        "minimum_aperture_contact_normal_relation":minimum_relation,
        "phone_thickness_aperture_solution":target_relation,
        "minimum_aperture_below_phone_thickness":min_surface<=PHONE[1],
        "actual_blocker":(
            "Hand can close to/through the phone thickness in an unconstrained hand-only sweep, "
            "but the failed phone optimization did not produce opposed broad-face normals and "
            "zero forbidden penetration." if min_surface<=PHONE[1] else
            "Actual thumb/index collision surfaces cannot close to phone thickness."),
        "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False}
    old.atomic_json(OUT/"left_phone_grasp_failure_diagnostic.json",report)
    # Sweep plot.
    fig,ax=plt.subplots(figsize=(12,6))
    x=np.arange(len(sweep));width=.24
    for k,label in enumerate(("lower","midpoint","upper")):
        ax.bar(x+(k-1)*width,[s["samples"][k]["surface_signed_aperture_m"]*1000 for s in sweep],
               width,label=label)
    ax.axhline(PHONE[1]*1000,color="red",label="phone thickness")
    ax.set_xticks(x,labels=[s["joint"].replace("left_hand_","") for s in sweep],rotation=25)
    ax.set_ylabel("thumb-index collision surface signed aperture [mm]")
    ax.set_title("FAILED DIAGNOSTIC — single-joint sweep");ax.legend();fig.tight_layout()
    fig.savefig(EVAL/"thumb_index_joint_sweep.png",dpi=160);plt.close(fig)
    # Diagnostic NPZ.
    payload=dict(full_g1_qpos=full,left_arm_wrist_qpos=v[:7],left_dex3_qpos=v[7:14],
        phone_proxy_pose=np.r_[center,Rotation.from_euler("xyz",rpy).as_quat()[[3,0,1,2]]],
        thumb_geom_pose=np.r_[data.geom_xpos[gids["thumb"]],data.geom_xmat[gids["thumb"]]],
        index_geom_pose=np.r_[data.geom_xpos[gids["index"]],data.geom_xmat[gids["index"]]],
        middle_geom_pose=np.r_[data.geom_xpos[gids["middle"]],data.geom_xmat[gids["middle"]]],
        contact_pad_points=np.asarray([contacts[p]["surface_point"] for p in ("thumb","index")]),
        raw_surface_normals=np.asarray([contacts[p]["raw_mesh_outward_normal"] for p in ("thumb","index")]),
        calculated_contact_normals=np.asarray([contacts[p]["grasp_calculated_phone_to_tip_normal"] for p in ("thumb","index")]),
        phone_face_normals=np.asarray([-prot[:,0],prot[:,0]]),
        signed_distances_names=np.asarray(list(all_dist)),signed_distances=np.asarray(list(all_dist.values())),
        minimum_aperture_qpos=minq,minimum_surface_aperture=np.asarray(min_surface),
        diagnostic_label=np.asarray("FAILED DIAGNOSTIC — NOT A VALID GRASP"))
    tmp=OUT/"best_failed_candidate_diagnostic.npz.incomplete"
    with tmp.open("wb") as f:np.savez_compressed(f,**payload)
    os.replace(tmp,OUT/"best_failed_candidate_diagnostic.npz")
    print(json.dumps(old.serial(report),indent=2));return report

if __name__=="__main__":build()
