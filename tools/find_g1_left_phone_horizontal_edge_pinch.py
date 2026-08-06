#!/usr/bin/env python3
"""Calibrate ALOHA jaws and find a static horizontal Dex3 edge pinch."""
from __future__ import annotations
import csv,json,os,sys,tempfile
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
import refine_g1_dex3_static_phone_contact as ref
import diagnose_g1_left_phone_grasp_failure as diag
OUT=ROOT/"converted_runs/g1_left_phone_horizontal_edge_pinch"
ACTION=ROOT/"evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
ALOHA=Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
PHONE=np.array([.1496,.00795,.0715]);TOL=.0002
def atomic_npz(p,**kw):
 t=p.with_suffix(p.suffix+".incomplete")
 with t.open("wb") as f:np.savez_compressed(f,**kw)
 os.replace(t,p)
def horizontal_model():
 m,tmp=old.expanded_phone_model(np.array([.3,.15,.9]),[0,0,0]);xml=Path(tmp)/"horizontal.xml"
 src=Path(tmp)/"g1_phone_proxy.xml";s=src.read_text()
 s=s.replace(f'size="{PHONE[1]/2} {PHONE[2]/2} {PHONE[0]/2}"',
             f'size="{PHONE[0]/2} {PHONE[2]/2} {PHONE[1]/2}"')
 xml.write_text(s);return mujoco.MjModel.from_xml_path(str(xml))
def main():
 OUT.mkdir(parents=True,exist_ok=True);act=np.load(ACTION)["optimized_action"].astype(float)
 # ALOHA FK and automatic gripper phases.
 am=mujoco.MjModel.from_xml_path(str(ALOHA));ad=mujoco.MjData(am)
 jids=[mujoco.mj_name2id(am,mujoco.mjtObj.mjOBJ_JOINT,f"follower_left_joint_{i}") for i in range(6)]
 qids=[am.jnt_qposadr[j] for j in jids]
 gj=mujoco.mj_name2id(am,mujoco.mjtObj.mjOBJ_JOINT,"follower_left_left_carriage_joint");gq=am.jnt_qposadr[gj]
 g=act[:,6];lo,hi=np.quantile(g,[.15,.85]);close_thr=lo+.35*(hi-lo);open_thr=lo+.65*(hi-lo)
 closed=False;labels=[];trans=[]
 for i,x in enumerate(g):
  oldc=closed
  if not closed and x<=close_thr:closed=True
  elif closed and x>=open_thr:closed=False
  if oldc!=closed:trans.append(i)
  labels.append("HOLD/MOVE" if closed else "APPROACH")
 # Contact frame at first open->closed transition after a genuinely open interval.
 cf=next((i for i in trans if i>10 and g[i]<=close_thr),int(np.argmin(g)))
 ad.qpos[qids]=act[cf,:6];ad.qpos[gq]=np.clip(g[cf],0,.044);mujoco.mj_forward(am,ad)
 ga=mujoco.mj_name2id(am,mujoco.mjtObj.mjOBJ_GEOM,"follower_left_gripper_right_tip")
 gb=mujoco.mj_name2id(am,mujoco.mjtObj.mjOBJ_GEOM,"follower_left_gripper_left_tip")
 pp=np.vstack((ad.geom_xpos[ga],ad.geom_xpos[gb]));closing=pp[0]-pp[1];closing/=np.linalg.norm(closing)
 tool=mujoco.mj_name2id(am,mujoco.mjtObj.mjOBJ_BODY,"follower_left_link_6");approach=ad.xmat[tool].reshape(3,3)[:,0]
 forward=approach-closing*np.dot(approach,closing);forward/=np.linalg.norm(forward);normal=np.cross(forward,closing)
 AR=np.column_stack((approach,closing,normal));ap=pp.mean(0)
 aloha_cal={"source":str(ACTION),"key":"optimized_action","shape":act.shape,
  "left_joint_names":[f"follower_left_joint_{i}" for i in range(6)],
  "gripper_joint":"follower_left_left_carriage_joint","jaw_bodies":["follower_left_carriage_right","follower_left_carriage_left"],
  "jaw_contact_geoms":["follower_left_gripper_right_tip","follower_left_gripper_left_tip"],
  "jaw_joint_axes":[[0,-1,0],[0,1,0]],"gripper_scalar_range_m":[float(g.min()),float(g.max())],
  "hysteresis":{"close_threshold_m":close_thr,"open_threshold_m":open_thr,"transition_frames_detected":trans},
  "contact_frame":cf,"contact_origin":ap,"rotation_columns_approach_closing_normal":AR,
  "jaw_opening_at_contact_m":float(np.linalg.norm(pp[0]-pp[1]))}
 old.atomic_json(OUT/"aloha_left_gripper_frame_calibration.json",aloha_cal)
 # G1 fixed finger branch; arm is solved so closing pads face +/- world Z.
 info=old.relative.latest.ik.validate_model(old.G1_XML);layout,_=old.hand_layout(info)
 natural=old.relative.load_natural_start(old.NATURAL_NPZ,info)
 seed=np.asarray(json.loads((ROOT/"converted_runs/g1_left_phone_cgap_grasp/left_dex3_cgap_seed.json").read_text())["joint_values"])
 model=horizontal_model();data=mujoco.MjData(model);pb=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"phone_proxy")
 phone=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"phone_proxy_geom")
 td=ref.collision_geoms(model,"left_hand_thumb_2_link")[-1];ix=ref.collision_geoms(model,"left_hand_index_1_link")[-1]
 wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_yaw_link")
 right=natural["arm_q"][7:];rhand=np.array([0.,-.2,-.3,.2,.3,.2,.3])
 def assign(q,pc=np.array([.3,.15,.9])):
  data.qpos[:]=model.key_qpos[0];data.qpos[info["arm_qpos_ids"][:7]]=q;data.qpos[info["arm_qpos_ids"][7:]]=right
  data.qpos[layout["hands"]["left"]["qadr"]]=seed;data.qpos[layout["hands"]["right"]["qadr"]]=rhand
  model.body_pos[pb]=pc;model.body_quat[pb]=[1,0,0,0];mujoco.mj_forward(model,data)
 assign(natural["arm_q"][:7]);_,ft=ref.distance(model,data,td,ix);pts=ft.reshape(2,3)
 ns=[np.asarray(diag.actual_surface(model,data,gid,p)["raw_mesh_outward_normal"]) for gid,p in ((td,pts[0]),(ix,pts[1]))]
 wR=data.xmat[wrist].reshape(3,3);lp=[wR.T@(p-data.xpos[wrist]) for p in pts];ln=[wR.T@n for n in ns]
 closing_g=pts[0]-pts[1];closing_g/=np.linalg.norm(closing_g)
 approach_g=wR[:,0]-closing_g*np.dot(wR[:,0],closing_g);approach_g/=np.linalg.norm(approach_g)
 normal_g=np.cross(approach_g,closing_g);GR0=np.column_stack((approach_g,closing_g,normal_g))
 gcal0={"joint_names":layout["hands"]["left"]["names"],"joint_values":seed,
  "pinch_origin":pts.mean(0),"rotation_columns_approach_closing_normal":GR0,
  "aperture_m":float(ref.distance(model,data,td,ix)[0]),"actual_normals":ns,
  "thumb_link":"left_hand_thumb_2_link","index_link":"left_hand_index_1_link",
  "middle_link":"left_hand_middle_1_link","wrist_frame":"left_wrist_yaw_link"}
 old.atomic_json(OUT/"g1_left_pinch_frame_calibration.json",gcal0)
 mapping0={"mapping":"ALOHA jaw A -> G1 thumb; jaw B -> G1 index",
  "T_aloha_gripper_to_g1_pinch_rotation":AR.T@GR0,"translation_tool_calibration":[0,0,0],
  "scope":"robot/tool embodiment calibration; independent of episode frame",
  "static_grasp_success_required_for_use":True}
 old.atomic_json(OUT/"aloha_to_g1_pinch_mapping.json",mapping0)
 lim=info["joint_limits"][:7];alo=lim[:,0]+.03;ahi=lim[:,1]-.03
 base=[natural["arm_q"][:7],np.array([-.67,.04,.25,1.38,.05,.06,-.03]),np.array([.2,.2,0,1.28,0,0,0])]
 rows=[];valid=[]
 for sign in (1,-1):
  for si,q0 in enumerate(base):
   q0=np.minimum(np.maximum(q0,alo),ahi)
   def fun(q):
    assign(q);R=data.xmat[wrist].reshape(3,3);p=data.xpos[wrist]
    return np.r_[8*(R@ln[0]-np.array([0,0,sign])),8*(R@ln[1]-np.array([0,0,-sign])),
      8*(p-np.array([.30,.15,.90])),1.2*(q[3]-1.25),.8*q[5],.8*q[6],.05*(q-natural["arm_q"][:7])]
   sol=least_squares(fun,q0,bounds=(alo,ahi),max_nfev=240)
   assign(sol.x);R=data.xmat[wrist].reshape(3,3);pad=np.asarray([data.xpos[wrist]+R@v for v in lp])
   # Edge grasp: contact midpoint lies 45-60 mm from phone center along long X.
   for edge in (-.060,-.050,-.040,.040,.050,.060):
    for lateral in (-.015,0,.015):
     pc=pad.mean(0)-np.array([edge,lateral,0]);assign(sol.x,pc)
     ds=[];norms=[];points=[];errs=[]
     for gid,target in ((td,np.array([0,0,sign])),(ix,np.array([0,0,-sign]))):
      dd,ff=ref.distance(model,data,gid,phone);ds.append(float(dd));points.append(ff[:3].copy())
      nn=np.asarray(diag.actual_surface(model,data,gid,ff[:3])["raw_mesh_outward_normal"]);norms.append(nn)
      errs.append(float(np.degrees(np.arccos(np.clip(np.dot(nn,target),-1,1)))))
     forbidden=[]
     for gg in range(model.ngeom):
      if gg in (phone,td,ix) or not (model.geom_contype[gg] or model.geom_conaffinity[gg]):continue
      if old.category(old.body_name(model,gg)) in ("finger","hand_wrist","arm"):
       dd,_=ref.distance(model,data,gg,phone);forbidden.append(float(dd))
     counts=0
     for c in data.contact:
      a,b=old.body_name(model,c.geom1),old.body_name(model,c.geom2)
      intended=("phone_proxy" in (a,b) and ("left_hand_thumb_2_link" in (a,b) or "left_hand_index_1_link" in (a,b)))
      if not intended and (("torso_link" in (a,b)) or (a.startswith("left_") and b.startswith("right_")) or (a.startswith("right_") and b.startswith("left_"))):counts+=1
     ap2,_=ref.distance(model,data,td,ix);nd=float(np.dot(norms[0],norms[1]));fm=min(forbidden,default=1.)
     margin=float(np.minimum(sol.x-lim[:,0],lim[:,1]-sol.x).min())
     ok=bool(abs(ap2-PHONE[1])<=.001 and nd<=-.95 and max(errs)<20 and all(-TOL<=d<=.0005 for d in ds)
      and fm>=-TOL and counts==0 and margin>=.03)
     rec={"candidate":len(rows),"sign":sign,"seed":si,"edge_offset":edge,"lateral_offset":lateral,
      "valid":ok,"aperture":ap2,"normal_dot":nd,"max_normal_error":max(errs),
      "thumb_distance":ds[0],"index_distance":ds[1],"forbidden_clearance":fm,"joint_margin":margin,
      "_q":sol.x.copy(),"_pc":pc.copy(),"_points":np.asarray(points),"_norms":np.asarray(norms)}
     rows.append(rec)
     if ok:valid.append(rec)
 fields=[k for k in rows[0] if not k.startswith("_")]
 with (OUT/"horizontal_edge_pinch_candidates.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fields);w.writeheader()
  for r in rows:w.writerow({k:r[k] for k in fields})
 if not valid:
  best=min(rows,key=lambda r:(max(abs(r["thumb_distance"]),abs(r["index_distance"]))+max(0,-r["forbidden_clearance"]),r["max_normal_error"]))
  report={"verdict":"G1_LEFT_HORIZONTAL_EDGE_PINCH_BLOCKED","candidate_count":len(rows),
   "blocker":"horizontal finite-phone contact/collision conflict","best_failed":{k:v for k,v in best.items() if not k.startswith("_")}}
  old.atomic_json(OUT/"horizontal_edge_pinch_report.json",report);print(json.dumps(report,indent=2));return 2
 best=max(valid,key=lambda r:(r["forbidden_clearance"],r["joint_margin"],-r["max_normal_error"]))
 assign(best["_q"],best["_pc"]);full=data.qpos.copy();GR=np.column_stack((np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,best["sign"]])))
 gcal={"joint_names":layout["hands"]["left"]["names"],"joint_values":seed,"pinch_origin":best["_points"].mean(0),
  "rotation_columns_approach_edge_closing":GR,"aperture_m":best["aperture"],"actual_normals":best["_norms"]}
 old.atomic_json(OUT/"g1_left_pinch_frame_calibration.json",gcal)
 calR=AR.T@GR;mapping={"mapping":"ALOHA jaw A -> G1 thumb; jaw B -> G1 index",
  "T_aloha_gripper_to_g1_pinch_rotation":calR,"translation_tool_calibration":[0,0,0],
  "scope":"robot/tool embodiment calibration; not episode-frame correction"}
 old.atomic_json(OUT/"aloha_to_g1_pinch_mapping.json",mapping)
 atomic_npz(OUT/"selected_horizontal_edge_pinch.npz",full_g1_qpos=full,left_arm_qpos=best["_q"],
  left_dex3_qpos=seed,right_parked_qpos=right,right_parked_dex3_qpos=rhand,
  horizontal_phone_proxy_pose=np.r_[best["_pc"],1,0,0,0],aloha_gripper_contact_frame=np.r_[ap,AR.ravel()],
  g1_pinch_contact_frame=np.r_[best["_points"].mean(0),GR.ravel()],contact_points=best["_points"],
  contact_normals=best["_norms"],aperture=np.asarray(best["aperture"]),
  collision_clearances=np.asarray([best["forbidden_clearance"]]),joint_margins=np.asarray([best["joint_margin"]]))
 report={"verdict":"G1_LEFT_HORIZONTAL_EDGE_PINCH_READY","candidate_count":len(rows),"valid_count":len(valid),
  "selected":{k:v for k,v in best.items() if not k.startswith("_")},"phone_dimensions_m":PHONE,
  "aloha_contact_frame":cf,"phase_transitions_detected":trans,"optimized_action_used":str(ACTION),
  "trajectory_generated":False,"physics_executed":False}
 old.atomic_json(OUT/"horizontal_edge_pinch_report.json",report);print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 0
if __name__=="__main__":raise SystemExit(main())
