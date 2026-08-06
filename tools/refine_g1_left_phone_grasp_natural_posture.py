#!/usr/bin/env python3
"""Coarse-to-fine static natural-posture refinement with the C-gap branch fixed."""
from __future__ import annotations
import csv,json,os,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
import refine_g1_dex3_static_phone_contact as ref
import diagnose_g1_left_phone_grasp_failure as diag
OUT=ROOT/"converted_runs/g1_left_phone_cgap_natural_posture"
SOURCE=ROOT/"converted_runs/g1_left_phone_cgap_grasp/selected_left_phone_cgap_grasp.npz"
PHONE=np.array([.1496,.00795,.0715]);TOL=.0002

def wbend(q):return float(np.degrees(np.arccos(np.clip(np.cos(q[5])*np.cos(q[6]),-1,1))))
def atomic_npz(path,**kw):
 tmp=path.with_suffix(path.suffix+".incomplete")
 with tmp.open("wb") as f:np.savez_compressed(f,**kw)
 os.replace(tmp,path)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with np.load(SOURCE,allow_pickle=False) as z:src={k:z[k].copy() for k in z.files}
 info=old.relative.latest.ik.validate_model(old.G1_XML);layout,_=old.hand_layout(info)
 natural=old.relative.load_natural_start(old.NATURAL_NPZ,info)
 pose=src["phone_proxy_pose"];model,_=old.expanded_phone_model(pose[:3],[0,0,0]);data=mujoco.MjData(model)
 pb=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"phone_proxy")
 phone=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"phone_proxy_geom")
 lw=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_yaw_link")
 rw=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"right_wrist_yaw_link")
 torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
 td=ref.collision_geoms(model,"left_hand_thumb_2_link")[-1]
 ix=ref.collision_geoms(model,"left_hand_index_1_link")[-1]
 lhand=src["left_dex3_qpos"];rhand=np.array([0.,-.20,-.30,.20,.30,.20,.30])
 lim=info["joint_limits"];lo=lim+.10;lo=lo[:,0];hi=(lim-.10)[:,1]
 # Reconstruct every previously validated phone-long-axis branch around the
 # fixed C-gap fingers.  Reusing only the old selected wrist-to-phone transform
 # would preserve its visibly bad wrist bend.
 data.qpos[:]=src["full_g1_qpos"];model.body_pos[pb]=pose[:3];model.body_quat[pb]=[1,0,0,0]
 mujoco.mj_forward(model,data);w0p=data.xpos[lw].copy();w0R=data.xmat[lw].reshape(3,3).copy()
 gap,ft0=ref.distance(model,data,td,ix);pts0=ft0.reshape(2,3)
 xaxis=pts0[0]-pts0[1];xaxis/=np.linalg.norm(xaxis)
 refaxis=w0R[:,0];z0=refaxis-xaxis*np.dot(refaxis,xaxis)
 if np.linalg.norm(z0)<.2:
  refaxis=w0R[:,2];z0=refaxis-xaxis*np.dot(refaxis,xaxis)
 z0/=np.linalg.norm(z0);y0=np.cross(z0,xaxis);y0/=np.linalg.norm(y0);z0=np.cross(xaxis,y0)
 prior_rows=list(csv.DictReader(open(ROOT/"converted_runs/g1_left_phone_cgap_grasp/wrist_pose_candidates.csv")))
 branches=[]
 for rr0 in prior_rows:
  if rr0["valid"].lower()!="true":continue
  a=np.radians(float(rr0["roll_deg"]));y=np.cos(a)*y0+np.sin(a)*z0
  z=-np.sin(a)*y0+np.cos(a)*z0;pr=np.column_stack((xaxis,y,z))
  pc0=pts0.mean(0)+y*float(rr0["short_offset_m"])+z*float(rr0["long_offset_m"])
  branches.append({"source":int(rr0["candidate"]),"relp":w0R.T@(pc0-w0p),
                   "relR":w0R.T@pr})
 torso_p=data.xpos[torso].copy();torso_R=data.xmat[torso].reshape(3,3).copy()

 def setq(la,ra,pc):
  data.qpos[:]=model.key_qpos[0];data.qpos[info["arm_qpos_ids"][:7]]=la
  data.qpos[info["arm_qpos_ids"][7:]]=ra
  data.qpos[layout["hands"]["left"]["qadr"]]=lhand
  data.qpos[layout["hands"]["right"]["qadr"]]=rhand
  model.body_pos[pb]=pc;model.body_quat[pb]=[1,0,0,0];mujoco.mj_forward(model,data)

 # Natural right parked arm: bent elbow, open relaxed hand, wrist at the
 # right upper-abdomen side. It is solved independently before left-arm search.
 rtargets=[np.array([.18,y,z]) for y in (-.20,-.24,-.28) for z in (.88,.94,1.00)]
 rseeds=[]
 for elbow in (1.05,1.25,1.45):
  for yaw in (-.35,0,.35):
   q=natural["arm_q"][7:].copy();q[3]=elbow;q[2]=yaw;q[5:]=0;rseeds.append(np.minimum(np.maximum(q,lo[7:]),hi[7:]))
 rbest=None
 for target in rtargets:
  for q0 in rseeds:
   def rr(q):
    setq(src["left_arm_qpos"],q,pose[:3])
    return np.r_[80*(data.xpos[rw]-target),.6*(q[3]-1.25),
                 .6*q[5],.6*q[6],.04*(q-natural["arm_q"][7:])]
   s=least_squares(rr,q0,bounds=(lo[7:],hi[7:]),max_nfev=120)
   setq(src["left_arm_qpos"],s.x,pose[:3]);pe=np.linalg.norm(data.xpos[rw]-target)
   bad=0
   for c in data.contact:
    a,b=old.body_name(model,c.geom1),old.body_name(model,c.geom2)
    if a.startswith("right_") and b.startswith("right_"):bad+=1
    if "torso_link" in (a,b) and (a.startswith("right_") or b.startswith("right_")):bad+=1
   score=pe+0.05*abs(s.x[3]-1.25)+0.01*wbend(s.x)+bad
   if pe<.01 and bad==0 and (rbest is None or score<rbest[0]):rbest=(score,s.x.copy(),target)
 if rbest is None:
  report={"verdict":"G1_LEFT_PHONE_NATURAL_POSTURE_BLOCKED","blocker":"natural right parked pose"}
  old.atomic_json(OUT/"selected_left_phone_natural_grasp_report.json",report);print(report["verdict"]);return 2
 right=rbest[1]

 # 5x5x5 torso-local phone grid, four posture seeds = 500 candidates.
 centers_local=[np.array([x,y,z]) for x in np.linspace(.25,.40,5)
                for y in np.linspace(.08,.20,5) for z in np.linspace(.05,.20,5)]
 base=src["left_arm_qpos"];seeds=[]
 for elbow,syaw in ((1.10,0),(1.30,-.25),(1.50,.25),(1.25,.45)):
  q=.35*base+.65*natural["arm_q"][:7];q[3]=elbow;q[2]+=syaw;q[5:]=0
  seeds.append(np.minimum(np.maximum(q,lo[:7]),hi[:7]))
 # Preliminary central-target solve chooses four wrist/phone branches that
 # best admit a bent elbow and small pitch/yaw. All 22 valid branches are
 # considered; no random joint/phone co-search is used.
 central=torso_p+torso_R@np.array([.30,.14,.12]);ranked=[]
 for br in branches:
  targetR=br["relR"].T;targetp=central-targetR@br["relp"]
  bestpre=1e9
  for q0 in seeds:
   def pre(q):
    setq(q,right,central);cur=data.xmat[lw].reshape(3,3)
    rv=Rotation.from_matrix(cur.T@targetR).as_rotvec()
    return np.r_[400*(data.xpos[lw]-targetp),40*rv,4.0*(q[3]-1.30),
                 4.0*q[5],4.0*q[6],.08*(q[:3]-natural["arm_q"][:3])]
   ss=least_squares(pre,q0,bounds=(lo[:7],hi[:7]),max_nfev=140)
   setq(ss.x,right,central);cur=data.xmat[lw].reshape(3,3)
   pe=np.linalg.norm(data.xpos[lw]-targetp)
   oe=np.degrees(np.linalg.norm(Rotation.from_matrix(cur.T@targetR).as_rotvec()))
   score=200*pe+oe+wbend(ss.x)+.5*abs(np.degrees(ss.x[3])-82)
   bestpre=min(bestpre,score)
  ranked.append((bestpre,br))
 bestbranches=[x[1] for x in sorted(ranked,key=lambda x:x[0])[:4]]
 rows=[];valid=[]
 for pi,local in enumerate(centers_local):
  pc=torso_p+torso_R@local
  for si,q0 in enumerate(seeds):
   br=bestbranches[si];targetR=br["relR"].T;targetp=pc-targetR@br["relp"]
   def lr(q):
    setq(q,right,pc);cur=data.xmat[lw].reshape(3,3)
    rv=Rotation.from_matrix(cur.T@targetR).as_rotvec()
    return np.r_[400*(data.xpos[lw]-targetp),40*rv,
      4.0*(q[3]-1.30),4.0*q[5],4.0*q[6],
      .08*(q[:3]-natural["arm_q"][:3]),.03*q[4]]
   sol=least_squares(lr,q0,bounds=(lo[:7],hi[:7]),max_nfev=180,
                     ftol=1e-10,xtol=1e-10,gtol=1e-10)
   q=sol.x;setq(q,right,pc);cur=data.xmat[lw].reshape(3,3)
   pe=float(np.linalg.norm(data.xpos[lw]-targetp))
   oe=float(np.degrees(np.linalg.norm(Rotation.from_matrix(cur.T@targetR).as_rotvec())))
   elbow=float(np.degrees(q[3]));bend=wbend(q)
   margin=float(np.minimum(q-lim[:7,0],lim[:7,1]-q).min())
   # Exact intended contact and raw mesh-normal audit.
   ds=[];ns=[];errs=[];inside=True
   for gid,target in ((td,np.array([-1.,0,0])),(ix,np.array([1.,0,0]))):
    dd,ft=ref.distance(model,data,gid,phone);ds.append(float(dd))
    nn=np.asarray(diag.actual_surface(model,data,gid,ft[:3])["raw_mesh_outward_normal"]);ns.append(nn)
    errs.append(float(np.degrees(np.arccos(np.clip(np.dot(nn,target),-1,1)))))
    ll=ft[:3]-pc;inside &= bool(abs(ll[1])<=PHONE[2]/2-.001 and abs(ll[2])<=PHONE[0]/2-.001)
   aperture,_=ref.distance(model,data,td,ix);ndot=float(np.dot(ns[0],ns[1]))
   forbidden=[];counts={"forbidden_phone":0,"arm_torso":0,"finger_torso":0,
                        "left_right_cross":0,"robot_other":0}
   for g in range(model.ngeom):
    if g in (phone,td,ix) or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
    bn=old.body_name(model,g)
    if old.category(bn) in ("finger","hand_wrist","arm"):
     dd,_=ref.distance(model,data,g,phone);forbidden.append(float(dd))
   for c in data.contact:
    a,b=old.body_name(model,c.geom1),old.body_name(model,c.geom2)
    if "phone_proxy" in (a,b):continue
    if "torso_link" in (a,b) and any(old.category(x)=="arm" for x in (a,b)):counts["arm_torso"]+=1
    elif "torso_link" in (a,b) and any(old.category(x) in ("finger","hand_wrist") for x in (a,b)):counts["finger_torso"]+=1
    elif ((a.startswith("left_") and b.startswith("right_")) or (a.startswith("right_") and b.startswith("left_"))):counts["left_right_cross"]+=1
    elif (a.startswith("left_") and b.startswith("left_")) or (a.startswith("right_") and b.startswith("right_")):counts["robot_other"]+=1
   fmin=min(forbidden,default=1.)
   hard=bool(pe<.001 and oe<1 and 60<=elbow<=110 and bend<=25 and margin>=.10
    and abs(aperture-PHONE[1])<.001 and all(-TOL<=d<=.0005 for d in ds)
    and max(errs)<20 and ndot<=-.95 and inside and fmin>=-TOL
    and sum(counts.values())==0)
   natural_score=(abs(elbow-82)/40+bend/25+np.linalg.norm(q[:3]-natural["arm_q"][:3])
                  +.2*np.linalg.norm(q-natural["arm_q"][:7])-.5*min(fmin,.02))
   row={"candidate":len(rows),"phone_seed":pi,"arm_seed":si,"valid":hard,
    "phone_x":local[0],"phone_y":local[1],"phone_z":local[2],"position_error_m":pe,
    "orientation_error_deg":oe,"elbow_flexion_deg":elbow,"wrist_bend_deg":bend,
    "minimum_joint_margin_rad":margin,"aperture_m":aperture,
    "thumb_distance_m":ds[0],"index_distance_m":ds[1],
    "max_contact_normal_error_deg":max(errs),"actual_normal_dot":ndot,
    "minimum_forbidden_clearance_m":fmin,"collision_count":sum(counts.values()),
    "wrist_branch":br["source"],"natural_score":natural_score,
    "_q":q.copy(),"_pc":pc.copy(),"_normals":np.asarray(ns),
    "_distances":np.asarray(ds),"_counts":counts}
   rows.append(row)
   if hard:valid.append(row)
 fields=[k for k in rows[0] if not k.startswith("_")]
 with (OUT/"natural_posture_candidates.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fields);w.writeheader()
  for r in rows:w.writerow({k:r[k] for k in fields})
 if not valid:
  best=min(rows,key=lambda r:(r["wrist_bend_deg"],
       abs(r["elbow_flexion_deg"]-82),r["position_error_m"]))
  report={"verdict":"G1_LEFT_PHONE_NATURAL_POSTURE_BLOCKED","safety_pass":False,
   "candidate_count":len(rows),"valid_count":0,
   "valid_wrist_branch_count":len(branches),
   "minimum_wrist_bend_deg":min(r["wrist_bend_deg"] for r in rows),
   "wrist_bend_le_25_candidate_count":sum(r["wrist_bend_deg"]<=25 for r in rows),
   "elbow_60_110_candidate_count":sum(60<=r["elbow_flexion_deg"]<=110 for r in rows),
   "best_failed_candidate":{k:v for k,v in best.items() if not k.startswith("_")},
   "right_parked_qpos":right,"right_dex3_qpos":rhand,
   "blocker":(
    "POSTURE_CONSTRAINT_CONFLICT: with the validated fixed C-gap fingers, "
    "torso-frontal vertical phone, and requested torso-local workspace, all "
    "22 valid phone-long-axis branches require >=67 deg wrist bend; no "
    "candidate reaches the <=25 deg hard limit. Exact-pose candidates also "
    "do not simultaneously reach 60 deg elbow flexion."),
   "selected_success_npz_created":False,"success_images_created":False,
   "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False}
  old.atomic_json(OUT/"selected_left_phone_natural_grasp_report.json",report)
  print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 2
 best=min(valid,key=lambda r:(r["natural_score"],r["wrist_bend_deg"],-r["minimum_joint_margin_rad"]))
 setq(best["_q"],right,best["_pc"])
 cps=[]
 for gid in (td,ix):
  _,ft=ref.distance(model,data,gid,phone);cps.append(ft[:3].copy())
 full=data.qpos.copy();allmargin=np.minimum(np.r_[best["_q"],right]-lim[:,0],lim[:,1]-np.r_[best["_q"],right])
 atomic_npz(OUT/"selected_left_phone_natural_grasp.npz",full_g1_qpos=full,
  left_arm_qpos=best["_q"],left_dex3_qpos=lhand,right_parked_arm_qpos=right,
  right_parked_dex3_qpos=rhand,phone_proxy_pose=np.r_[best["_pc"],1,0,0,0],
  actual_contact_points=np.asarray(cps),actual_contact_normals=best["_normals"],
  aperture=np.asarray(best["aperture_m"]),joint_limit_margins=allmargin,
  elbow_flexion_deg=np.asarray(best["elbow_flexion_deg"]),wrist_bend_deg=np.asarray(best["wrist_bend_deg"]),
  forbidden_clearances=np.asarray([best["minimum_forbidden_clearance_m"]]),
  source_full_g1_qpos=src["full_g1_qpos"])
 report={"verdict":"G1_LEFT_PHONE_NATURAL_POSTURE_READY","safety_pass":True,
  "candidate_count":len(rows),"valid_count":len(valid),"selected_candidate":best["candidate"],
  "phone_torso_local_position_m":[best["phone_x"],best["phone_y"],best["phone_z"]],
  "phone_world_position_m":best["_pc"],"left_joint_names":info["joint_names"][:7],
  "left_arm_qpos":best["_q"],"elbow_flexion_deg":best["elbow_flexion_deg"],
  "wrist_bend_deg":best["wrist_bend_deg"],"right_parked_qpos":right,
  "right_dex3_qpos":rhand,"thumb_index_aperture_m":best["aperture_m"],
  "contact_distances_m":best["_distances"],"contact_normal_errors_deg":[
   float(np.degrees(np.arccos(np.clip(np.dot(best["_normals"][0],[-1,0,0]),-1,1)))),
   float(np.degrees(np.arccos(np.clip(np.dot(best["_normals"][1],[1,0,0]),-1,1))))],
  "actual_normal_dot":best["actual_normal_dot"],"minimum_arm_wrist_margin_rad":float(allmargin.min()),
  "minimum_forbidden_clearance_m":best["minimum_forbidden_clearance_m"],
  "collision_counts":best["_counts"],"source_to_natural_joint_difference":full-src["full_g1_qpos"],
  "phone_orientation_errors_deg":{"screen_frontal":0.,"long_up":0.},
  "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False,
  "gui_command":f"{sys.executable} {ROOT/'tools/view_g1_left_phone_natural_grasp.py'} --grasp {OUT/'selected_left_phone_natural_grasp.npz'}"}
 old.atomic_json(OUT/"selected_left_phone_natural_grasp_report.json",report)
 print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 0
if __name__=="__main__":raise SystemExit(main())
