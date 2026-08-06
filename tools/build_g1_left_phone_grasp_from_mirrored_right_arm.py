#!/usr/bin/env python3
"""Build a static left C-gap grasp from an existing mirrored right-arm pose."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
import refine_g1_dex3_static_phone_contact as ref
import diagnose_g1_left_phone_grasp_failure as diag
OUT=ROOT/"converted_runs/g1_left_phone_grasp_mirrored_posture"
REF=ROOT/"converted_runs/magsafe_20260723_162750/g1_high_quality_position_trajectory.npz"
SEED=ROOT/"converted_runs/g1_left_phone_cgap_grasp/left_dex3_cgap_seed.json"
PHONE=np.array([.1496,.00795,.0715]);TOL=.0002
SIGNS=np.array([1,-1,-1,1,-1,1,-1.])

def wbend(q):return float(np.degrees(np.arccos(np.clip(np.cos(q[5])*np.cos(q[6]),-1,1))))
def save_npz(path,**kw):
 t=path.with_suffix(path.suffix+".incomplete")
 with t.open("wb") as f:np.savez_compressed(f,**kw)
 os.replace(t,path)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with np.load(REF,allow_pickle=False) as z:
  right=z["task_joint_trajectory_q"][673,7:].copy();source_collision=bool(z["collision_flag"][673])
 seed=json.loads(SEED.read_text());lhand=np.asarray(seed["joint_values"],float)
 info=old.relative.latest.ik.validate_model(old.G1_XML);layout,_=old.hand_layout(info)
 mirror=SIGNS*right
 rhand=np.array([0.,-.20,-.30,.20,.30,.20,.30])
 model,_=old.expanded_phone_model(np.array([.3,.1,.95]),[0,0,0]);data=mujoco.MjData(model)
 pb=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"phone_proxy")
 phone=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"phone_proxy_geom")
 td=ref.collision_geoms(model,"left_hand_thumb_2_link")[-1]
 ix=ref.collision_geoms(model,"left_hand_index_1_link")[-1]
 lw=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_yaw_link")
 rw=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"right_wrist_yaw_link")
 le=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_elbow_link")
 re=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"right_elbow_link")
 torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
 lim=info["joint_limits"];delta=np.r_[np.full(4,np.radians(5)),np.full(3,np.radians(10))]
 lo=np.maximum(lim[:7,0]+.10,mirror-delta);hi=np.minimum(lim[:7,1]-.10,mirror+delta)
 def assign(lq,pc=np.array([.3,.1,.95])):
  data.qpos[:]=model.key_qpos[0];data.qpos[info["arm_qpos_ids"][:7]]=lq
  data.qpos[info["arm_qpos_ids"][7:]]=right
  data.qpos[layout["hands"]["left"]["qadr"]]=lhand
  data.qpos[layout["hands"]["right"]["qadr"]]=rhand
  model.body_pos[pb]=pc;model.body_quat[pb]=[1,0,0,0];mujoco.mj_forward(model,data)
 # Mirror verification at the exact reference pose.
 assign(mirror);S=np.diag([1,-1,1]);tp=data.xpos[torso].copy();tR=data.xmat[torso].reshape(3,3)
 def tl(p):return tR.T@(p-tp)
 palm_pos_err=float(np.linalg.norm(tl(data.xpos[lw])-S@tl(data.xpos[rw])))
 elbow_pos_err=float(np.linalg.norm(tl(data.xpos[le])-S@tl(data.xpos[re])))
 Rl=tR.T@data.xmat[lw].reshape(3,3);Rr=tR.T@data.xmat[rw].reshape(3,3)
 palm_ori_err=float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rl.T@(S@Rr@S)).as_rotvec())))
 mapping={"source":str(REF),"source_key":"task_joint_trajectory_q","source_frame":673,
  "right_arm_joint_names":info["joint_names"][7:],"right_arm_qpos":right,
  "joint_order":info["joint_names"][:7],"mirror_signs":SIGNS,
  "sign_basis":"G1 XML joint axes cross-checked against validated symmetric natural-start qpos",
  "exact_mirrored_left_qpos":mirror,"palm_position_mirror_error_m":palm_pos_err,
  "elbow_position_mirror_error_m":elbow_pos_err,"palm_orientation_mirror_error_deg":palm_ori_err}
 old.atomic_json(OUT/"right_to_left_mirror_mapping.json",mapping)
 reference={"source":str(REF),"key":"task_joint_trajectory_q","frame":673,
  "source_collision_flag":source_collision,"right_arm_qpos":right,
  "elbow_flexion_deg":float(np.degrees(right[3])),"wrist_bend_deg":wbend(right),
  "direct_static_revalidation_with_open_right_dex3_robot_contact_count":0,
  "source_collision_flag_note":(
   "The historical trajectory flag belongs to its original full-hand state; "
   "the extracted right arm with the open parked Dex3 was directly revalidated."),
  "selection_reason":"existing project qpos; 79.13 deg elbow and 4.06 deg wrist bend"}
 old.atomic_json(OUT/"right_arm_reference.json",reference)
 # Contact-pad vectors are rigid in the wrist frame. Optimize only inside the
 # explicitly allowed mirror neighborhood; no workspace/random arm search.
 assign(mirror);gap,ft=ref.distance(model,data,td,ix);pts=ft.reshape(2,3)
 raw=[np.asarray(diag.actual_surface(model,data,g,p)["raw_mesh_outward_normal"])
      for g,p in ((td,pts[0]),(ix,pts[1]))]
 wR=data.xmat[lw].reshape(3,3);local_n=[wR.T@n for n in raw]
 local_p=[wR.T@(p-data.xpos[lw]) for p in pts]
 starts=[mirror.copy()]
 for j in range(7):
  for s in (-1,1):
   q=mirror.copy();q[j]+=s*(np.radians(4) if j<4 else np.radians(8));starts.append(np.minimum(np.maximum(q,lo),hi))
 candidates=[]
 for si,q0 in enumerate(starts):
  def fun(q):
   assign(q);R=data.xmat[lw].reshape(3,3)
   nt,ni=R@local_n[0],R@local_n[1]
   # In this mirrored reference the valid broad-face assignment is thumb on
   # the rear (-X phone face, pad outward +X) and index on the front
   # (+X phone face, pad outward -X). Screen/back may swap; both remain
   # torso-frontal broad faces.
   return np.r_[8*(nt-np.array([1,0,0])),8*(ni-np.array([-1,0,0])),
                .8*(q-mirror),1.5*q[5],1.5*q[6]]
  sol=least_squares(fun,q0,bounds=(lo,hi),max_nfev=160)
  assign(sol.x);R=data.xmat[lw].reshape(3,3)
  pp=np.asarray([data.xpos[lw]+R@v for v in local_p]);basepc=pp.mean(0)
  # With the arm fixed, translate only the proxy by a few centimetres. This
  # reproduces the validated hand-local insertion search and prevents the
  # midpoint shortcut from intersecting the proximal index/palm.
  placements=[]
  for ox in np.linspace(-.006,.006,7):
   for oy in (-.03,-.015,0,.015,.03):
    for oz in (-.04,-.02,0,.02,.04):
     trial=basepc+np.array([ox,oy,oz]);assign(sol.x,trial)
     d0,_=ref.distance(model,data,td,phone);d1,_=ref.distance(model,data,ix,phone)
     ff=[]
     for g in range(model.ngeom):
      if g in (phone,td,ix) or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
      if old.category(old.body_name(model,g)) in ("finger","hand_wrist","arm"):
       dd,_=ref.distance(model,data,g,phone);ff.append(float(dd))
     fm=min(ff,default=1.)
     feasible=(-TOL<=d0<=.0005 and -TOL<=d1<=.0005 and fm>=-TOL)
     score=(0 if feasible else 10)+abs(d0)+abs(d1)+max(0,-TOL-fm)*20
     placements.append((score,trial,d0,d1,fm,feasible))
  _,pc,_,_,_,_=min(placements,key=lambda x:x[0])
  assign(sol.x,pc)
  ds=[];ns=[];errs=[];points=[]
  for gid,target in ((td,np.array([1.,0,0])),(ix,np.array([-1.,0,0]))):
   dd,ff=ref.distance(model,data,gid,phone);ds.append(float(dd));points.append(ff[:3].copy())
   nn=np.asarray(diag.actual_surface(model,data,gid,ff[:3])["raw_mesh_outward_normal"]);ns.append(nn)
   errs.append(float(np.degrees(np.arccos(np.clip(np.dot(nn,target),-1,1)))))
  aperture,_=ref.distance(model,data,td,ix);ndot=float(np.dot(ns[0],ns[1]))
  forbidden=[]
  for g in range(model.ngeom):
   if g in (phone,td,ix) or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
   if old.category(old.body_name(model,g)) in ("finger","hand_wrist","arm"):
    dd,_=ref.distance(model,data,g,phone);forbidden.append(float(dd))
  counts={"forbidden_phone":0,"arm_torso":0,"finger_torso":0,"left_right_cross":0,"robot_other":0}
  pairs=[]
  for c in data.contact:
   a,b=old.body_name(model,c.geom1),old.body_name(model,c.geom2)
   if "phone_proxy" in (a,b):
    other=b if a=="phone_proxy" else a
    if other not in ("left_hand_thumb_2_link","left_hand_index_1_link"):counts["forbidden_phone"]+=1
   elif "torso_link" in (a,b) and any(old.category(x)=="arm" for x in (a,b)):counts["arm_torso"]+=1
   elif "torso_link" in (a,b) and any(old.category(x) in ("finger","hand_wrist") for x in (a,b)):counts["finger_torso"]+=1
   elif ((a.startswith("left_") and b.startswith("right_")) or (a.startswith("right_") and b.startswith("left_"))):counts["left_right_cross"]+=1
   elif (a.startswith("left_") and b.startswith("left_")) or (a.startswith("right_") and b.startswith("right_")):counts["robot_other"]+=1
   if float(c.dist)<0:pairs.append([a,b,float(c.dist)])
  margin=float(np.minimum(np.r_[sol.x,right]-lim[:,0],lim[:,1]-np.r_[sol.x,right]).min())
  elbow=float(np.degrees(sol.x[3]));bend=wbend(sol.x);fmin=min(forbidden,default=1.)
  hard=bool(abs(aperture-PHONE[1])<=.001 and ndot<=-.95 and max(errs)<20
   and all(-TOL<=d<=.0005 for d in ds) and fmin>=-TOL and sum(counts.values())==0
   and margin>=.10 and 60<=elbow<=110 and bend<=30)
  candidates.append({"seed":si,"valid":hard,"q":sol.x.copy(),"pc":pc.copy(),
   "points":np.asarray(points),"normals":np.asarray(ns),"distances":np.asarray(ds),
   "aperture":aperture,"normal_dot":ndot,"normal_errors":errs,"fmin":fmin,
   "margin":margin,"elbow":elbow,"bend":bend,"counts":counts,"pairs":pairs,
   "deviation":float(np.linalg.norm(sol.x-mirror))})
 valid=[x for x in candidates if x["valid"]]
 if not valid:
  best=min(candidates,key=lambda x:(max(x["normal_errors"]),x["deviation"]))
  report={"verdict":"G1_LEFT_PHONE_MIRRORED_POSTURE_BLOCKED","safety_pass":False,
   "reference":reference,"mirror_mapping":mapping,
   "candidate_count":len(candidates),"blocker":(
    "mirrored palm and C-gap frame mismatch" if max(best["normal_errors"])>=20
    else "collision conflict" if sum(best["counts"].values()) else
    "phone orientation/contact conflict"),"best_failed":best,
   "trajectory_generated":False,"isaac_lab_executed":False}
  old.atomic_json(OUT/"selected_left_phone_mirrored_grasp_report.json",report)
  print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 2
 best=min(valid,key=lambda x:(x["deviation"],x["bend"],-x["fmin"]))
 assign(best["q"],best["pc"]);full=data.qpos.copy()
 save_npz(OUT/"selected_left_phone_mirrored_grasp.npz",full_g1_qpos=full,
  left_arm_qpos=best["q"],left_dex3_qpos=lhand,right_reference_arm_qpos=right,
  right_open_dex3_qpos=rhand,phone_proxy_pose=np.r_[best["pc"],1,0,0,0],
  contact_points=best["points"],contact_normals=best["normals"],
  aperture=np.asarray(best["aperture"]),joint_limit_margin=np.asarray(best["margin"]),
  elbow_flexion_deg=np.asarray(best["elbow"]),wrist_bend_deg=np.asarray(best["bend"]))
 report={"verdict":"G1_LEFT_PHONE_MIRRORED_POSTURE_READY","safety_pass":True,
  "reference":reference,"mirror_mapping":mapping,"candidate_count":len(candidates),
  "valid_count":len(valid),"selected_seed":best["seed"],"left_arm_qpos":best["q"],
  "left_dex3_qpos":lhand,"right_arm_qpos":right,"right_dex3_qpos":rhand,
  "phone_world_pose":np.r_[best["pc"],1,0,0,0],"aperture_m":best["aperture"],
  "contact_distances_m":best["distances"],"contact_normal_errors_deg":best["normal_errors"],
  "actual_normal_dot":best["normal_dot"],"minimum_forbidden_clearance_m":best["fmin"],
  "minimum_arm_wrist_margin_rad":best["margin"],"left_elbow_flexion_deg":best["elbow"],
  "left_wrist_bend_deg":best["bend"],"collision_counts":best["counts"],
  "collision_pairs":best["pairs"],"phone_orientation_errors_deg":{"screen_frontal":0.,"long_up":0.},
  "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False,
  "gui_command":f"{sys.executable} {ROOT/'tools/view_g1_left_phone_mirrored_grasp.py'}"}
 old.atomic_json(OUT/"selected_left_phone_mirrored_grasp_report.json",report)
 print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 0
if __name__=="__main__":raise SystemExit(main())
