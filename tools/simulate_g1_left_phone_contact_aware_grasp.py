#!/usr/bin/env python3
"""Contact-aware Dex3 phone grasp using the fixed validated PREGRASP."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import mujoco,numpy as np
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import simulate_g1_left_phone_physics_grasp as base
import find_g1_dex3_static_phone_grasp as old
import refine_g1_dex3_static_phone_contact as ref
import diagnose_g1_left_phone_grasp_failure as diagmod
OUT=ROOT/"evaluation/g1_left_phone_contact_aware_grasp"
PRIM=ROOT/"converted_runs/g1_left_phone_contact_aware_grasp"
SRC=ROOT/"evaluation/g1_left_phone_physics_grasp/physics_grasp_report.json"
PAD_MODE=False
PAD_CAL_OUT=ROOT/"evaluation/g1_left_phone_pad_contact_calibration"

def copy_robot_key(src,model,data):
 for oj in range(src.njnt):
  n=mujoco.mj_id2name(src,mujoco.mjtObj.mjOBJ_JOINT,oj)
  nj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
  if nj<0:continue
  typ=src.jnt_type[oj];w=7 if typ==mujoco.mjtJoint.mjJNT_FREE else 4 if typ==mujoco.mjtJoint.mjJNT_BALL else 1
  a=src.jnt_qposadr[oj];b=model.jnt_qposadr[nj];data.qpos[b:b+w]=src.key_qpos[0,a:a+w]

def run_case(cfg,src_report,info,layout,src_model):
 mu=cfg["friction"];model=base.wrapper(np.asarray(src_report["phone_proxy_position_m"]),mu);data=mujoco.MjData(model)
 copy_robot_key(src_model,model,data)
 arm_names=[mujoco.mj_id2name(src_model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in info["arm_joint_ids"]]
 hand_names=layout["hands"]["left"]["names"]+layout["hands"]["right"]["names"]
 def jq(n):return model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]
 aq=np.array([jq(n) for n in arm_names]);hq=np.array([jq(n) for n in hand_names])
 pre=np.asarray(src_report["pregrasp_qpos"]);close=np.asarray(src_report["close_target_qpos"])
 qarm=np.asarray(src_report["arm_pregrasp_qpos"]);qlift=np.asarray(src_report["arm_lift_10cm_ik_qpos"])
 right=np.array([.2,-.2,0,1.28,0,0,0]);rhand=np.array([0.,-.2,-.3,.2,.3,.2,.3])
 data.qpos[aq[:7]]=qarm;data.qpos[aq[7:]]=right;data.qpos[hq[:7]]=pre;data.qpos[hq[7:]]=rhand
 fj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"phone_free");fq=model.jnt_qposadr[fj]
 pc=np.asarray(src_report["phone_proxy_position_m"]);data.qpos[fq:fq+7]=np.r_[pc,1,0,0,0];mujoco.mj_forward(model,data)
 names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
 amap={n:i for i,n in enumerate(names)}
 thumb_names=layout["hands"]["left"]["names"][:3];index_names=layout["hands"]["left"]["names"][5:]
 thumb_act=[amap[n] for n in thumb_names];index_act=[amap[n] for n in index_names]
 if cfg["force_limit"] is not None:
  for a in thumb_act+index_act:
   model.actuator_forcelimited[a]=1;model.actuator_forcerange[a]=[-cfg["force_limit"],cfg["force_limit"]]
 def ctrl(arm,hand):
  for n,v in zip(arm_names[:7],arm):data.ctrl[amap[n]]=v
  for n,v in zip(arm_names[7:],right):data.ctrl[amap[n]]=v
  for n,v in zip(hand_names[:7],hand):data.ctrl[amap[n]]=v
  for n,v in zip(hand_names[7:],rhand):data.ctrl[amap[n]]=v
 phone=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"physics_phone_geom")
 tg=ref.collision_geoms(model,"left_hand_thumb_2_link")[-1]
 i0g=ref.collision_geoms(model,"left_hand_index_0_link")[-1]
 ig=ref.collision_geoms(model,"left_hand_index_1_link")[-1]
 pad_geoms={"thumb":tg,"index0":i0g,"index1":ig}
 # Calibrate inner-pad triangle sets in the unchanged PREGRASP frame.
 pad_triangles={};pad_cal={};pad_mesh={}
 for label,gid in pad_geoms.items():
  mid=int(model.geom_dataid[gid]);fa=int(model.mesh_faceadr[mid]);fn=int(model.mesh_facenum[mid])
  va=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid])
  verts=np.asarray(model.mesh_vert[va:va+vn]);faces=np.asarray(model.mesh_face[fa:fa+fn])
  G=data.geom_xmat[gid].reshape(3,3);gp=data.geom_xpos[gid]
  world=(G@verts.T).T+gp;tri=world[faces]
  normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);areas=np.linalg.norm(normals,axis=1)/2
  normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
  cent=tri.mean(1);toward=np.sign(pc[2]-cent[:,2])[:,None]*np.array([0.,0.,1.])
  allowed=np.where((np.sum(normals*toward,axis=1)>np.cos(np.deg2rad(30)))&
                   (areas>np.percentile(areas,15)))[0]
  pad_triangles[label]=set(map(int,allowed))
  local_tri=verts[faces];local_norm=np.cross(local_tri[:,1]-local_tri[:,0],local_tri[:,2]-local_tri[:,0])
  local_norm/=np.maximum(np.linalg.norm(local_norm,axis=1,keepdims=True),1e-12)
  pad_mesh[label]=(local_tri.mean(1),local_norm)
  pad_cal[label]={"body":old.body_name(model,gid),"geom_id":gid,"mesh_id":mid,
   "triangle_count":fn,"allowed_inner_pad_triangle_indices":allowed.tolist(),
   "normal_alignment_threshold_deg":30.,"edge_area_percentile_excluded":15.}
 dt=model.opt.timestep
 delta=(close-pre);base_duration=6.;duration=base_duration/max(cfg["speed"],1e-9)
 T=max(16.,duration+7.);N=int(T/dt);target=pre.copy();arm_target=qarm.copy()
 state={"thumb":"CLOSING","index":"CLOSING"};stop_q={"thumb":None,"index":None};contact_time=None
 logs=[];poses=[];pens=[];valid_counts={"thumb":0,"index":0};sat_steps=0
 max_inst=0.;pair_max={};forbidden_patch_max=0.;bilateral_steps=0;last_bilateral=False;initial_z=pc[2]
 for k in range(N):
  t=k*dt
  if cfg["controller"]=="baseline":
   u=np.clip((t-1)/base_duration,0,1);target=pre+u*delta
  else:
   u=np.clip((t-1)/duration,0,1)
   proposed=pre+u*delta
   if state["thumb"]=="CLOSING":target[:3]=proposed[:3]
   if state["index"]=="CLOSING":target[5:]=proposed[5:]
  if contact_time is not None:
   tau=t-contact_time
   if .75<tau<=2.75:arm_target=qarm+min(1,(tau-.75)/1.0)*.5*(qlift-qarm)
   elif 2.75<tau<=4.75:arm_target=qarm+(.5+.5*min(1,(tau-2.75)/1.0))*(qlift-qarm)
  ctrl(arm_target,target);mujoco.mj_step(model,data)
  th=ix=False;tf=xf=0.;step_pen=[];step_pairs=[]
  for c in data.contact:
   gs={c.geom1,c.geom2}
   if phone not in gs:continue
   other=c.geom2 if c.geom1==phone else c.geom1
   body=old.body_name(model,other);pen=max(0,-float(c.dist))
   if body and body.startswith(("left_hand_","right_hand_")):
    step_pen.append(pen);key=f"phone|{body}";pair_max[key]=max(pair_max.get(key,0),pen);step_pairs.append(key)
   force=float(data.efc_force[c.efc_address]) if c.efc_address>=0 else 0.
   # Intended contact is a calibrated triangle patch, not an entire link.
   def is_pad(label,gid):
    if other!=gid:return False
    G=data.geom_xmat[gid].reshape(3,3);local=G.T@(np.asarray(c.pos)-data.geom_xpos[gid])
    centers,local_normals=pad_mesh[label];tri=int(np.argmin(np.sum((centers-local)**2,axis=1)));n=G@local_normals[tri]
    R=data.geom_xmat[phone].reshape(3,3);local=R.T@(np.asarray(c.pos)-data.geom_xpos[phone])
    face_inside=abs(local[0])<=base.PHONE[0]/2-.002 and abs(local[1])<=base.PHONE[2]/2-.002
    face_n=np.sign(data.geom_xpos[phone,2]-c.pos[2])*np.array([0.,0.,1.])
    return tri in pad_triangles[label] and np.dot(n,face_n)>=np.cos(np.deg2rad(30)) and face_inside
   if PAD_MODE:
    thumb_pad=is_pad("thumb",tg);index_pad=is_pad("index0",i0g) or is_pad("index1",ig)
    if thumb_pad:th=True;tf=max(tf,force)
    if index_pad:ix=True;xf=max(xf,force)
    if body and body.startswith(("left_hand_","right_hand_")) and not (thumb_pad or index_pad):
     forbidden_patch_max=max(forbidden_patch_max,pen)
   else:
    if other==tg:th=True;tf=max(tf,force)
    if other==ig:ix=True;xf=max(xf,force)
  if cfg["controller"]!="baseline":
   if th and state["thumb"]=="CLOSING":
    state["thumb"]="FORCE_HOLD";target[:3]=data.qpos[hq[:3]];stop_q["thumb"]=target[:3].copy()
   if ix and state["index"]=="CLOSING":
    state["index"]="FORCE_HOLD";target[5:]=data.qpos[hq[5:7]];stop_q["index"]=target[5:7].copy()
  if th:valid_counts["thumb"]+=1
  if ix:valid_counts["index"]+=1
  bilateral=th and ix
  if bilateral:
   bilateral_steps+=1
   if contact_time is None:contact_time=t
  last_bilateral=bilateral
  p=max(step_pen,default=0.);max_inst=max(max_inst,p)
  af=data.actuator_force[thumb_act+index_act]
  if cfg["force_limit"] is not None and np.any(np.abs(af)>=.98*cfg["force_limit"]):sat_steps+=1
  q=data.qpos[hq[:7]].copy();qd=np.array([data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]] for n in hand_names[:7]])
  logs.append([t,*target,*q,*qd,*af,tf,xf])
  poses.append([t,*data.qpos[fq:fq+7]]);pens.append([t,p,*[pair_max.get(x,0) for x in sorted(pair_max)]])
 logs=np.asarray(logs);poses=np.asarray(poses)
 pen_series=np.asarray([r[1] for r in pens]);w10=max(1,int(.010/dt));w50=max(1,int(.050/dt))
 sustained=lambda w:float(max((pen_series[i:i+w].min() for i in range(max(1,len(pen_series)-w+1))),default=0))
 lift=float(poses[:,3].max()-initial_z);drop=bool(poses[-1,3]<initial_z-.02)
 if PAD_MODE:
  forbidden=max(forbidden_patch_max,max((v for k,v in pair_max.items() if not any(x in k for x in ("thumb_2_link","index_0_link","index_1_link"))),default=0.))
 else:
  forbidden=max((v for k,v in pair_max.items() if "thumb_2_link" not in k and "index_1_link" not in k),default=0.)
 success=bool(contact_time is not None and lift>=.05 and not drop and sustained(w10)<.001 and forbidden<=1e-6 and last_bilateral and sat_steps*dt<.1)
 rec={**cfg,"success":success,"contact_formed":contact_time is not None,"contact_time_s":contact_time,
  "thumb_contact_duration_s":valid_counts["thumb"]*dt,"index_contact_duration_s":valid_counts["index"]*dt,
  "lift_height_m":lift,"drop":drop,"instantaneous_max_penetration_m":max_inst,
  "sustained_10ms_penetration_m":sustained(w10),"sustained_50ms_penetration_m":sustained(w50),
  "forbidden_penetration_m":forbidden,"pair_max_penetration_m":pair_max,
  "actuator_saturation_duration_s":sat_steps*dt,"thumb_stop_qpos":stop_q["thumb"],"index_stop_qpos":stop_q["index"]}
 return rec,logs,poses,np.asarray(pens,dtype=object),pad_cal

def main():
 OUT.mkdir(parents=True,exist_ok=True);PRIM.mkdir(parents=True,exist_ok=True)
 src=json.loads(SRC.read_text());info=old.relative.latest.ik.validate_model(old.G1_XML);layout,_=old.hand_layout(info)
 sm=mujoco.MjModel.from_xml_path(str(old.G1_XML))
 finger=["left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint","left_hand_index_0_joint","left_hand_index_1_joint"]
 acts={}
 for n in finger:
  a=mujoco.mj_name2id(sm,mujoco.mjtObj.mjOBJ_ACTUATOR,n)
  acts[n]={"type":"position","kp":float(sm.actuator_gainprm[a,0]),"kv":float(-sm.actuator_biasprm[a,2]),
   "ctrlrange":sm.actuator_ctrlrange[a].tolist(),"forcerange":sm.actuator_forcerange[a].tolist()}
 diag={"actuators":acts,"physics_timestep_s":float(sm.opt.timestep),"solver_iterations":int(sm.opt.iterations),
  "baseline_closing_duration_s":6.0,"baseline_target_speed_rad_s":(np.abs(np.asarray(src["close_target_qpos"])-np.asarray(src["pregrasp_qpos"]))/6).tolist(),
  "baseline_result":src["friction_results"],"interpretation":"kp=500 position targets have no XML force limit"}
 old.atomic_json(OUT/"baseline_controller_diagnosis.json",diag)
 configs=[]
 for mu in (.4,.7,1.0):
  configs.append({"controller":"baseline","speed":1.0,"force_level":"unlimited","force_limit":None,"friction":mu})
  for speed in ((.25,) if PAD_MODE else (.25,.1)):
   for label,force in ((("low",.5),("medium",1.0)) if PAD_MODE else (("low",.5),("medium",1.0),("high",2.0))):
    configs.append({"controller":"contact_aware","speed":speed,"force_level":label,"force_limit":force,"friction":mu})
 results=[];runs=[]
 for i,c in enumerate(configs):
  print(f"[{i+1}/{len(configs)}] {c}",flush=True);r,*a=run_case(c,src,info,layout,sm);results.append(r);runs.append(a)
 fields=[k for k in results[0] if k not in ("pair_max_penetration_m","thumb_stop_qpos","index_stop_qpos")]
 with (OUT/"controller_candidate_comparison.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in results])
 valid=[(i,r) for i,r in enumerate(results) if r["success"]]
 best_i,best=(min(valid,key=lambda x:(x[1]["sustained_10ms_penetration_m"],-x[1]["lift_height_m"])) if valid else
  min(enumerate(results),key=lambda x:(x[1]["sustained_10ms_penetration_m"]>=.001,-x[1]["lift_height_m"])))
 logs,poses,pens,pad_cal=runs[best_i]
 np.savez_compressed(OUT/"phone_pose_trajectory.npz",trajectory=poses,config=json.dumps(configs[best_i]))
 np.savetxt(OUT/"actuator_trajectory.csv",logs,delimiter=",")
 np.savetxt(OUT/"penetration_trajectory.csv",np.asarray([[x[0],x[1]] for x in pens],float),delimiter=",",header="time_s,max_penetration_m")
 np.savetxt(OUT/"contact_force_trajectory.csv",logs[:,[0,-2,-1]],delimiter=",",header="time_s,thumb_force_n,index_force_n")
 # Exact 100 ms controller trace after close begins for the baseline diagnosis.
 blog=runs[0][0];bp=runs[0][2]
 mask=(blog[:,0]>=1.0)&(blog[:,0]<=1.1)
 diag["close_first_100ms_columns"]=["time_s",*[f"target_q{i}" for i in range(7)],
  *[f"actual_q{i}" for i in range(7)],*[f"joint_velocity{i}" for i in range(7)],
  *[f"actuator_force{i}" for i in range(5)],"thumb_contact_force_n","index_contact_force_n",
  "signed_contact_distance_proxy_m"]
 diag["close_first_100ms_trace"]=[[*row.tolist(),-float(p[1])] for row,p in zip(blog[mask],bp[mask])]
 old.atomic_json(OUT/"baseline_controller_diagnosis.json",diag)
 verdict=("G1_LEFT_PHONE_PAD_CONTACT_GRASP_READY" if valid and PAD_MODE else
  "G1_LEFT_PHONE_CONTACT_AWARE_GRASP_READY" if valid else
  "PAD_ALIGNMENT_FAILED" if PAD_MODE and not best["contact_formed"] else
  "INDEX_LATERAL_GEOMETRY_COLLISION" if best["forbidden_penetration_m"]>0 else "PENETRATION_REMAINS_EXCESSIVE")
 report={"verdict":verdict,"candidate_count":len(results),"all_results":results,"selected":best,
  "new_posture_search_performed":False,"phone_welded":False,"isaac_lab_executed":False,
  "videos_generated":False,"gui_command":f"python {__file__} --gui"}
 old.atomic_json(OUT/"contact_aware_grasp_report.json",report)
 if PAD_MODE:
  PAD_CAL_OUT.mkdir(parents=True,exist_ok=True)
  old.atomic_json(PAD_CAL_OUT/"dex3_pad_contact_calibration.json",{
   "definition":"inner pad triangles: outward normal within 30 deg of phone-facing broad-face normal; smallest 15% triangles excluded as mesh-edge/joint regions",
   "patches":pad_cal,"whole_link_is_not_automatically_intended":True})
  import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
  for label,name in (("thumb","thumb_pad_triangles.png"),("index0","index0_inner_pad_triangles.png"),("index1","index1_inner_pad_triangles.png")):
   n=pad_cal[label]["triangle_count"];allowed=set(pad_cal[label]["allowed_inner_pad_triangle_indices"])
   fig,ax=plt.subplots(figsize=(10,2));ax.scatter(range(n),[1 if i in allowed else 0 for i in range(n)],
    c=["tab:green" if i in allowed else "tab:red" for i in range(n)],s=5)
   ax.set(title=f"{label}: calibrated mesh triangles",xlabel="mesh triangle index",ylabel="inner pad");fig.tight_layout();fig.savefig(PAD_CAL_OUT/name,dpi=180);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,3))
  ax.bar(pad_cal.keys(),[len(x["allowed_inner_pad_triangle_indices"]) for x in pad_cal.values()],color="tab:green",label="allowed")
  ax.bar(pad_cal.keys(),[x["triangle_count"]-len(x["allowed_inner_pad_triangle_indices"]) for x in pad_cal.values()],
   bottom=[len(x["allowed_inner_pad_triangle_indices"]) for x in pad_cal.values()],color="tab:red",label="forbidden")
  ax.legend();ax.set_ylabel("triangle count");fig.tight_layout();fig.savefig(PAD_CAL_OUT/"allowed_vs_forbidden_contact_regions.png",dpi=180);plt.close(fig)
 if valid:
  old.atomic_json(PRIM/"dex3_phone_contact_aware_primitive.json",{"OPEN":"existing natural open","PREGRASP":src["pregrasp_qpos"],
   "controller":"PREGRASP -> independent contact detection -> force hold","selected":best,"validated_friction_results":[r for r in results if r["success"]]})
 print(json.dumps(old.serial({"verdict":verdict,"selected":best}),indent=2));return 0 if valid else 2
if __name__=="__main__":raise SystemExit(main())
