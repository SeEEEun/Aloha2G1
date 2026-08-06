#!/usr/bin/env python3
"""Deterministic Dex3 finger-manifold search on the fixed mirrored palm."""
from __future__ import annotations
import csv,json,os,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.stats import qmc
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
import refine_g1_dex3_static_phone_contact as ref
import diagnose_g1_left_phone_grasp_failure as diag
OUT=ROOT/"converted_runs/g1_left_phone_mirrored_cgap"
MREPORT=ROOT/"converted_runs/g1_left_phone_grasp_mirrored_posture/selected_left_phone_mirrored_grasp_report.json"
PHONE=np.array([.1496,.00795,.0715]);TOL=.0002;N=20480

def wbend(q):return float(np.degrees(np.arccos(np.clip(np.cos(q[5])*np.cos(q[6]),-1,1))))
def npz(path,**kw):
 t=path.with_suffix(path.suffix+".incomplete")
 with t.open("wb") as f:np.savez_compressed(f,**kw)
 os.replace(t,path)

def main():
 OUT.mkdir(parents=True,exist_ok=True);mr=json.loads(MREPORT.read_text())
 arm=np.asarray(mr["best_failed"]["q"],float);right=np.asarray(mr["reference"]["right_arm_qpos"],float)
 rhand=np.array([0.,-.20,-.30,.20,.30,.20,.30])
 info=old.relative.latest.ik.validate_model(old.G1_XML);layout,_=old.hand_layout(info)
 ranges=layout["hands"]["left"]["ranges"];flo=ranges[:,0]+.03;fhi=ranges[:,1]-.03
 model,_=old.expanded_phone_model(np.array([.3,.1,.9]),[0,0,0]);data=mujoco.MjData(model)
 pb=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"phone_proxy")
 phone=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"phone_proxy_geom")
 td=ref.collision_geoms(model,"left_hand_thumb_2_link")[-1];ix=ref.collision_geoms(model,"left_hand_index_1_link")[-1]
 md=ref.collision_geoms(model,"left_hand_middle_1_link")[-1]
 bodies={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in
  ("torso_link","left_elbow_link","left_wrist_roll_link","left_wrist_yaw_link")}
 def assign(a,h,pc=np.array([.3,.1,.9])):
  data.qpos[:]=model.key_qpos[0];data.qpos[info["arm_qpos_ids"][:7]]=a
  data.qpos[info["arm_qpos_ids"][7:]]=right;data.qpos[layout["hands"]["left"]["qadr"]]=h
  data.qpos[layout["hands"]["right"]["qadr"]]=rhand;model.body_pos[pb]=pc
  model.body_quat[pb]=[1,0,0,0];mujoco.mj_forward(model,data)
 # Reconstruct and save the arm before changing any finger value.
 seed=np.asarray(json.loads((ROOT/"converted_runs/g1_left_phone_cgap_grasp/left_dex3_cgap_seed.json").read_text())["joint_values"])
 assign(arm,seed)
 frames={}
 for n,b in bodies.items():frames[n]=np.r_[data.xpos[b],data.xmat[b].reshape(3,3).ravel()]
 fullref=data.qpos.copy();marg=np.minimum(np.r_[arm,right]-info["joint_limits"][:,0],
                                         info["joint_limits"][:,1]-np.r_[arm,right])
 npz(OUT/"mirrored_natural_arm_reference.npz",full_g1_qpos=fullref,left_arm_qpos=arm,
  right_parked_arm_qpos=right,right_parked_dex3_qpos=rhand,frame_names=np.asarray(list(frames)),
  frame_poses=np.asarray(list(frames.values())),elbow_flexion_deg=np.asarray(np.degrees(arm[3])),
  wrist_bend_deg=np.asarray(wbend(arm)),arm_wrist_joint_margins=marg)
 # Deterministic low-discrepancy manifold search. Middle joints are biased
 # toward the folded half of their legal ranges without fixing one pose.
 u=qmc.Sobol(7,scramble=False).random_base2(15)[:N]
 q=flo+(fhi-flo)*u
 q[:,3:5]=flo[3:5]+(fhi[3:5]-flo[3:5])*(.55+.45*u[:,3:5])
 fields=["candidate","aperture_m","aperture_pass","normal_dot","normal_pass",
  "thumb_normal_x","index_normal_x","face_alignment_error_deg","middle_gap_clearance_m",
  "finger_margin_rad","manifold_valid"]
 csvpath=OUT/"mirrored_palm_finger_manifold.csv"
 resume_ids=[]
 if csvpath.exists():
  with csvpath.open() as f:
   cached=list(csv.DictReader(f))
  if len(cached)==N:
   resume_ids=[int(r["candidate"]) for r in cached if r["manifold_valid"].lower()=="true"]
 rows=[];manifold=[]
 for i in ([] if resume_ids else range(N)):
  h=q[i]
  assign(arm,h);ap,ft=ref.distance(model,data,td,ix);apok=abs(ap-PHONE[1])<=.001
  rec={"candidate":i,"aperture_m":ap,"aperture_pass":apok,"normal_dot":np.nan,
   "normal_pass":False,"thumb_normal_x":np.nan,"index_normal_x":np.nan,
   "face_alignment_error_deg":np.nan,"middle_gap_clearance_m":np.nan,
   "finger_margin_rad":float(np.minimum(h-ranges[:,0],ranges[:,1]-h).min()),"manifold_valid":False}
  if apok:
   pts=ft.reshape(2,3);closing=pts[0]-pts[1];closing/=np.linalg.norm(closing)+1e-12
   coarse_face_err=float(np.degrees(np.arccos(np.clip(abs(closing[0]),-1,1))))
   # A raw mesh normal cannot rescue a closing axis already >8 degrees from
   # both legal broad-face normals. This cheap exact-geom gate avoids running
   # the Python triangle query on geometrically impossible aperture samples.
   if coarse_face_err>=8:
    rec["face_alignment_error_deg"]=coarse_face_err;rows.append(rec);continue
   ns=[np.asarray(diag.actual_surface(model,data,g,p)["raw_mesh_outward_normal"])
       for g,p in ((td,pts[0]),(ix,pts[1]))]
   nd=float(np.dot(ns[0],ns[1]));dmid,_=ref.distance(model,data,md,td)
   # Both screen-normal signs are legal; select the better broad-face assignment.
   ea=max(np.degrees(np.arccos(np.clip(np.dot(ns[0],[1,0,0]),-1,1))),
          np.degrees(np.arccos(np.clip(np.dot(ns[1],[-1,0,0]),-1,1))))
   eb=max(np.degrees(np.arccos(np.clip(np.dot(ns[0],[-1,0,0]),-1,1))),
          np.degrees(np.arccos(np.clip(np.dot(ns[1],[1,0,0]),-1,1))))
   err=float(min(ea,eb));valid=bool(nd<=-.95 and dmid>.003 and rec["finger_margin_rad"]>=.03)
   rec.update(normal_dot=nd,normal_pass=nd<=-.95,thumb_normal_x=ns[0][0],
    index_normal_x=ns[1][0],face_alignment_error_deg=err,middle_gap_clearance_m=dmid,
    manifold_valid=valid)
   if valid:manifold.append({"i":i,"h":h.copy(),"pts":pts.copy(),"ns":np.asarray(ns),"err":err})
  rows.append(rec)
 if resume_ids:
  for i in resume_ids:
   h=q[i];assign(arm,h);_,ft=ref.distance(model,data,td,ix);pts=ft.reshape(2,3)
   ns=[np.asarray(diag.actual_surface(model,data,g,p)["raw_mesh_outward_normal"])
       for g,p in ((td,pts[0]),(ix,pts[1]))]
   ea=max(np.degrees(np.arccos(np.clip(np.dot(ns[0],[1,0,0]),-1,1))),
          np.degrees(np.arccos(np.clip(np.dot(ns[1],[-1,0,0]),-1,1))))
   eb=max(np.degrees(np.arccos(np.clip(np.dot(ns[0],[-1,0,0]),-1,1))),
          np.degrees(np.arccos(np.clip(np.dot(ns[1],[1,0,0]),-1,1))))
   manifold.append({"i":i,"h":h.copy(),"pts":pts.copy(),"ns":np.asarray(ns),"err":float(min(ea,eb))})
 else:
  with csvpath.open("w",newline="") as f:
   w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 # Evaluate the most torso-frontal manifold samples first. Phone is always
 # identity-oriented; only +/-5 mm translation is allowed.
 manifold.sort(key=lambda x:x["err"]);cont=[];valid=[]
 def evaluate(a,h,stage,source):
  assign(a,h);_,ft=ref.distance(model,data,td,ix);base=ft.reshape(2,3).mean(0)
  best=None
  for dx in np.linspace(-.005,.005,11):
   for dy in (0.,):
    for dz in (0.,):
     pc=base+np.array([dx,dy,dz]);assign(a,h,pc)
     ds=[];ns=[];points=[];errs=[]
     # infer assignment from x signs at actual phone surface
     for gid in (td,ix):
      dd,ff=ref.distance(model,data,gid,phone);ds.append(float(dd));points.append(ff[:3].copy())
      ns.append(np.asarray(diag.actual_surface(model,data,gid,ff[:3])["raw_mesh_outward_normal"]))
     for sign in (1,-1):
      ee=[np.degrees(np.arccos(np.clip(np.dot(ns[0],[sign,0,0]),-1,1))),
          np.degrees(np.arccos(np.clip(np.dot(ns[1],[-sign,0,0]),-1,1)))]
      if not errs or max(ee)<max(errs):errs=ee
     forbidden=[]
     for g in range(model.ngeom):
      if g in (phone,td,ix) or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
      if old.category(old.body_name(model,g)) in ("finger","hand_wrist","arm"):
       dd,_=ref.distance(model,data,g,phone);forbidden.append(float(dd))
     fmin=min(forbidden,default=1.);score=max(errs)+2000*sum(max(0,-TOL-d) for d in ds)+2000*max(0,-TOL-fmin)
     item=(score,pc,np.asarray(ds),np.asarray(ns),np.asarray(points),errs,fmin)
     if best is None or score<best[0]:best=item
  _,pc,ds,ns,points,errs,fmin=best;assign(a,h,pc)
  ap,_=ref.distance(model,data,td,ix);nd=float(np.dot(ns[0],ns[1]))
  counts={"forbidden_phone":0,"arm_torso":0,"finger_torso":0,"left_right_cross":0,"robot_other":0}
  for c in data.contact:
   x,y=old.body_name(model,c.geom1),old.body_name(model,c.geom2)
   if "phone_proxy" in (x,y):
    other=y if x=="phone_proxy" else x
    if other not in ("left_hand_thumb_2_link","left_hand_index_1_link"):counts["forbidden_phone"]+=1
   elif "torso_link" in (x,y) and any(old.category(v)=="arm" for v in (x,y)):counts["arm_torso"]+=1
   elif "torso_link" in (x,y) and any(old.category(v) in ("finger","hand_wrist") for v in (x,y)):counts["finger_torso"]+=1
   elif ((x.startswith("left_") and y.startswith("right_")) or (x.startswith("right_") and y.startswith("left_"))):counts["left_right_cross"]+=1
   elif (x.startswith("left_") and y.startswith("left_")) or (x.startswith("right_") and y.startswith("right_")):counts["robot_other"]+=1
  am=float(np.minimum(np.r_[a,right]-info["joint_limits"][:,0],info["joint_limits"][:,1]-np.r_[a,right]).min())
  fm=float(np.minimum(h-ranges[:,0],ranges[:,1]-h).min());el=float(np.degrees(a[3]));bend=wbend(a)
  hard=bool(abs(ap-PHONE[1])<=.001 and nd<=-.95 and max(errs)<20
   and all((-TOL<=d<=.0005) for d in ds) and fmin>=-TOL and sum(counts.values())==0
   and am>=.10 and fm>=.03 and 60<=el<=110 and bend<=30)
  return {"stage":stage,"source_candidate":source,"valid":hard,"arm":a.copy(),"hand":h.copy(),
   "pc":pc,"points":points,"normals":ns,"aperture":ap,"normal_dot":nd,"normal_errors":errs,
   "distances":ds,"fmin":fmin,"arm_margin":am,"finger_margin":fm,"elbow":el,"bend":bend,
   "counts":counts,"arm_deviation":float(np.linalg.norm(a-arm))}
 for m in manifold[:300]:cont.append(evaluate(arm,m["h"],1,m["i"]))
 valid=[x for x in cont if x["valid"]]
 # Limited continuation only on the best manifold samples.
 if not valid and manifold:
  for stage,sd,wd in ((2,2,5),(3,5,10)):
   da=np.r_[np.full(4,np.radians(sd)),np.full(3,np.radians(wd))]
   alo=np.maximum(info["joint_limits"][:7,0]+.10,arm-da);ahi=np.minimum(info["joint_limits"][:7,1]-.10,arm+da)
   for m in manifold[:30]:
    # Align mutual pad normals to the nearer +/- torso X assignment.
    h=m["h"];assign(arm,h);R0=data.xmat[bodies["left_wrist_yaw_link"]].reshape(3,3)
    lp=[R0.T@n for n in m["ns"]]
    sign=1 if abs(m["ns"][0][0]-1)+abs(m["ns"][1][0]+1)<abs(m["ns"][0][0]+1)+abs(m["ns"][1][0]-1) else -1
    def rr(a):
     assign(a,h);R=data.xmat[bodies["left_wrist_yaw_link"]].reshape(3,3)
     return np.r_[6*(R@lp[0]-np.array([sign,0,0])),6*(R@lp[1]-np.array([-sign,0,0])),.8*(a-arm)]
    sol=least_squares(rr,arm,bounds=(alo,ahi),max_nfev=100)
    cont.append(evaluate(sol.x,h,stage,m["i"]))
   valid=[x for x in cont if x["valid"]]
   if valid:break
 # Persist exact deterministic Stage-2/3 states for failed-collision visual
 # diagnosis. These files are never treated as selected/success grasps.
 def save_failed_diagnostic(x,path):
  assign(x["arm"],x["hand"],x["pc"])
  forbidden_points=[];arm_torso_points=[];pairs=[]
  for cc in data.contact:
   a,b=old.body_name(model,cc.geom1),old.body_name(model,cc.geom2)
   intended=("phone_proxy" in (a,b) and
             (("left_hand_thumb_2_link" in (a,b)) or
              ("left_hand_index_1_link" in (a,b))))
   if float(cc.dist)<0 and not intended:
    forbidden_points.append(np.asarray(cc.pos).copy())
    pairs.append(f"{a} <-> {b} : {float(cc.dist):.12g} m")
   if "torso_link" in (a,b) and any(old.category(v)=="arm" for v in (a,b)):
    arm_torso_points.append(np.asarray(cc.pos).copy())
  npz(path,diagnostic_label=np.asarray("FAILED COLLISION DIAGNOSTIC — NOT A VALID GRASP"),
   stage=np.asarray(x["stage"]),full_g1_qpos=data.qpos.copy(),left_arm_qpos=x["arm"],
   left_dex3_qpos=x["hand"],right_parked_qpos=right,right_parked_dex3_qpos=rhand,
   phone_proxy_pose=np.r_[x["pc"],1,0,0,0],intended_contact_points=x["points"],
   actual_contact_normals=x["normals"],intended_signed_distances=x["distances"],
   aperture=np.asarray(x["aperture"]),normal_dot=np.asarray(x["normal_dot"]),
   normal_errors_deg=np.asarray(x["normal_errors"]),
   minimum_forbidden_clearance_m=np.asarray(x["fmin"]),
   forbidden_penetration_points=np.asarray(forbidden_points).reshape(-1,3),
   arm_torso_collision_points=np.asarray(arm_torso_points).reshape(-1,3),
   collision_pairs=np.asarray(pairs),collision_counts_json=np.asarray(json.dumps(x["counts"])),
   elbow_flexion_deg=np.asarray(x["elbow"]),wrist_bend_deg=np.asarray(x["bend"]))
 stage2=[x for x in cont if x["stage"]==2]
 stage3=[x for x in cont if x["stage"]==3]
 if stage2:save_failed_diagnostic(stage2[-1],OUT/"best_stage2_collision_diagnostic.npz")
 if stage3:save_failed_diagnostic(stage3[-1],OUT/"best_stage3_collision_diagnostic.npz")
 # Local fine search on the single discovered manifold. The arm remains at
 # the already-computed Stage-2/3 continuation solutions; only finger q moves
 # inside a small deterministic neighborhood.
 if not valid and manifold:
  baseh=manifold[0]["h"];u2=qmc.Sobol(7,scramble=False).random_base2(3)
  arm_seeds=[x for x in cont if x["stage"] in (2,3)][-2:]
  for ai,ac in enumerate(arm_seeds):
   for j,v in enumerate(u2):
    h=baseh+(v-.5)*.16
    if np.any(h<flo)|np.any(h>fhi):continue
    item=evaluate(ac["arm"],h,4+ai,manifold[0]["i"]);cont.append(item)
    if item["valid"]:valid.append(item)
   if valid:break
 cf=["stage","source_candidate","valid","aperture","normal_dot","normal_errors","distances","fmin",
     "arm_margin","finger_margin","elbow","bend","arm_deviation"]
 with (OUT/"continuation_candidates.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,cf);w.writeheader()
  for x in cont:w.writerow({k:(json.dumps(old.serial(x[k])) if isinstance(x[k],(list,np.ndarray)) else x[k]) for k in cf})
 if not valid:
  def fail_rank(x):
   aperture_bad=max(0,abs(x["aperture"]-PHONE[1])-.001)
   normal_bad=max(0,x["normal_dot"]+.95)+max(0,max(x["normal_errors"])-20)
   contact_bad=sum(max(0,-TOL-d)+max(0,d-.0005) for d in x["distances"])
   collision_bad=max(0,-TOL-x["fmin"])+sum(x["counts"].values())
   return (aperture_bad,normal_bad,contact_bad+collision_bad,x["arm_deviation"])
  best=min(cont,key=fail_rank) if cont else None
  reason=("MIRRORED_PALM_FINGER_MANIFOLD_CONFLICT" if not manifold else
          "PHONE_ORIENTATION_CONTACT_CONFLICT" if best and max(best["normal_errors"])>=20 else
          "COLLISION_CONFLICT")
  report={"verdict":"G1_LEFT_PHONE_MIRRORED_CGAP_BLOCKED","blocker":reason,
   "coarse_candidate_count":N,"manifold_valid_count":len(manifold),
   "continuation_candidate_count":len(cont),"best_failed":best,
   "trajectory_generated":False,"isaac_lab_executed":False}
  old.atomic_json(OUT/"selected_left_phone_mirrored_cgap_report.json",report)
  print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 2
 best=min(valid,key=lambda x:(x["arm_deviation"],x["bend"],max(x["normal_errors"]),-x["fmin"]))
 assign(best["arm"],best["hand"],best["pc"]);full=data.qpos.copy()
 frame_names=list(bodies);frame_poses=np.asarray([np.r_[data.xpos[bodies[n]],data.xmat[bodies[n]].reshape(3,3).ravel()] for n in frame_names])
 pinch=np.r_[best["points"].mean(0),np.eye(3).ravel()]
 npz(OUT/"selected_left_phone_mirrored_cgap.npz",full_g1_qpos=full,mirrored_reference_qpos=fullref,
  left_arm_qpos=best["arm"],left_dex3_qpos=best["hand"],right_parked_qpos=right,
  right_parked_dex3_qpos=rhand,phone_proxy_pose=np.r_[best["pc"],1,0,0,0],
  frame_names=np.asarray(frame_names+["pinch"]),frame_poses=np.vstack((frame_poses,pinch)),
  contact_points=best["points"],actual_contact_normals=best["normals"],aperture=np.asarray(best["aperture"]),
  elbow_flexion_deg=np.asarray(best["elbow"]),wrist_bend_deg=np.asarray(best["bend"]),
  joint_margins=np.r_[best["arm_margin"],best["finger_margin"]],
  collision_clearances=np.asarray([best["fmin"]]),mirrored_reference_joint_deviation=np.asarray(best["arm_deviation"]))
 report={"verdict":"G1_LEFT_PHONE_MIRRORED_CGAP_READY","safety_pass":True,
  "coarse_candidate_count":N,"manifold_valid_count":len(manifold),"continuation_candidate_count":len(cont),
  "selected_stage":best["stage"],"source_candidate":best["source_candidate"],
  "left_arm_qpos":best["arm"],"left_dex3_qpos":best["hand"],"phone_pose":np.r_[best["pc"],1,0,0,0],
  "aperture_m":best["aperture"],"actual_normal_dot":best["normal_dot"],
  "contact_normal_errors_deg":best["normal_errors"],"contact_distances_m":best["distances"],
  "minimum_forbidden_clearance_m":best["fmin"],"minimum_arm_wrist_margin_rad":best["arm_margin"],
  "minimum_finger_margin_rad":best["finger_margin"],"elbow_flexion_deg":best["elbow"],
  "wrist_bend_deg":best["bend"],"collision_counts":best["counts"],
  "mirrored_reference_joint_deviation_rad":best["arm_deviation"],
  "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False}
 old.atomic_json(OUT/"selected_left_phone_mirrored_cgap_report.json",report)
 print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 0
if __name__=="__main__":raise SystemExit(main())
