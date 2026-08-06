#!/usr/bin/env python3
"""Physics-in-the-loop horizontal phone pinch; no weld/mocap phone attachment."""
from __future__ import annotations
import csv,json,os,sys,tempfile
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
import refine_g1_dex3_static_phone_contact as ref
OUT=ROOT/"evaluation/g1_left_phone_physics_grasp"
PRIM=ROOT/"converted_runs/g1_left_phone_physics_grasp"
PHONE=np.array([.1496,.00795,.0715]);FPS=30
def wrapper(phone_pos,friction):
 base=mujoco.MjModel.from_xml_path(str(old.G1_XML));td=Path(tempfile.mkdtemp(prefix="g1_phone_physics_"));p=td/"scene.xml"
 mujoco.mj_saveLastXML(str(p),base);s=p.read_text();assets=old.G1_XML.parent/"assets"
 s=s.replace('meshdir="assets/"',f'meshdir="{assets}/"')
 body=(f'<body name="physics_phone" pos="{" ".join(map(str,phone_pos))}"><freejoint name="phone_free"/>'
  f'<geom name="physics_phone_geom" type="box" size="{PHONE[0]/2} {PHONE[2]/2} {PHONE[1]/2}" '
  f'mass="0.20" friction="{friction} 0.005 0.0001" rgba=".05 .25 .95 1"/></body>'
  f'<geom name="physics_table" type="box" pos="{phone_pos[0]} {phone_pos[1]} {phone_pos[2]-PHONE[1]/2-.02}" '
  f'size=".35 .35 .02" friction="{friction} 0.005 0.0001" rgba=".5 .35 .2 1"/>')
 s=s.replace("<worldbody>","<worldbody>"+body,1)
 s=s.replace("</mujoco>",'<equality><weld name="fixed_g1_base" body1="pelvis"/></equality></mujoco>')
 p.write_text(s);return mujoco.MjModel.from_xml_path(str(p))
def main():
 OUT.mkdir(parents=True,exist_ok=True);PRIM.mkdir(parents=True,exist_ok=True)
 info=old.relative.latest.ik.validate_model(old.G1_XML);layout,_=old.hand_layout(info)
 natural=old.relative.load_natural_start(old.NATURAL_NPZ,info)
 seed=np.asarray(json.loads((ROOT/"converted_runs/g1_left_phone_cgap_grasp/left_dex3_cgap_seed.json").read_text())["joint_values"])
 # Recompute a horizontal-pinch wrist pose from the natural task-ready seed.
 # No qpos from the failed finite-phone static solution is loaded.
 qarm0=np.array([.2,.2,0,1.28,0,0,0]);right=np.array([.2,-.2,0,1.28,0,0,0])
 rhand=np.array([0.,-.2,-.3,.2,.3,.2,.3])
 km=mujoco.MjModel.from_xml_path(str(old.G1_XML));kd=mujoco.MjData(km)
 kd.qpos[:]=km.key_qpos[0];kd.qpos[info["arm_qpos_ids"][:7]]=qarm0;kd.qpos[info["arm_qpos_ids"][7:]]=right
 kd.qpos[layout["hands"]["left"]["qadr"]]=seed;kd.qpos[layout["hands"]["right"]["qadr"]]=rhand;mujoco.mj_forward(km,kd)
 tdg=ref.collision_geoms(km,"left_hand_thumb_2_link")[-1];ixg=ref.collision_geoms(km,"left_hand_index_1_link")[-1]
 _,ft=ref.distance(km,kd,tdg,ixg);pads0=ft.reshape(2,3)
 wrist=mujoco.mj_name2id(km,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_yaw_link")
 wR=kd.xmat[wrist].reshape(3,3);wp=kd.xpos[wrist].copy()
 local_pads=np.asarray([wR.T@(p-wp) for p in pads0])
 close_axis=(pads0[0]-pads0[1]);close_axis/=np.linalg.norm(close_axis)
 # Orient the calibrated closing axis to table normal and keep a natural elbow.
 local_close=wR.T@close_axis
 lim=info["joint_limits"][:7];alo=lim[:,0]+.03;ahi=lim[:,1]-.03
 def arm_residual(q):
  kd.qpos[info["arm_qpos_ids"][:7]]=q
  mujoco.mj_forward(km,kd)
  R=kd.xmat[wrist].reshape(3,3);p=kd.xpos[wrist]
  return np.r_[8*(R@local_close-np.array([0,0,1.])),
   7*(p-np.array([.30,.15,.90])),1.2*(q[3]-1.25),.8*q[5],.8*q[6],.08*(q-qarm0)]
 qarm=least_squares(arm_residual,np.clip(qarm0,alo,ahi),bounds=(alo,ahi),max_nfev=300).x
 kd.qpos[info["arm_qpos_ids"][:7]]=qarm
 kd.qpos[layout["hands"]["left"]["qadr"]]=seed;mujoco.mj_forward(km,kd)
 lift_R=kd.xmat[wrist].reshape(3,3).copy();lift_p=kd.xpos[wrist].copy()+np.array([0,0,.10])
 def lift_residual(q):
  kd.qpos[info["arm_qpos_ids"][:7]]=q;mujoco.mj_forward(km,kd)
  R=kd.xmat[wrist].reshape(3,3);p=kd.xpos[wrist]
  return np.r_[12*(p-lift_p),4*(R[:,0]-lift_R[:,0]),4*(R[:,2]-lift_R[:,2]),.12*(q-qarm)]
 qlift=least_squares(lift_residual,qarm,bounds=(alo,ahi),max_nfev=300).x
 kd.qpos[info["arm_qpos_ids"][:7]]=qarm;mujoco.mj_forward(km,kd)
 _,ft=ref.distance(km,kd,tdg,ixg);closed_pads=ft.reshape(2,3)
 # Put the finite phone at an edge-side grasp location, not at its centre.
 # X is the long phone axis; the pad midpoint is 50 mm from phone centre.
 pc=closed_pads.mean(0)-np.array([-.050,0,0])
 # Deterministically search a genuinely open finger branch around the validated
 # C-gap seed.  The failed complete static phone qpos is never referenced.
 hand_jids=[mujoco.mj_name2id(km,mujoco.mjtObj.mjOBJ_JOINT,n)
            for n in layout["hands"]["left"]["names"]]
 hlo=np.array([km.jnt_range[j,0] for j in hand_jids]);hhi=np.array([km.jnt_range[j,1] for j in hand_jids])
 sampler=qmc.Sobol(5,scramble=False)
 uv=sampler.random_base2(12)
 candidates=[]
 for u in uv:
  h=seed.copy()
  ids=np.array([0,1,2,5,6])
  span=np.minimum(.55*(hhi[ids]-hlo[ids]),np.array([.55,.65,.65,.65,.65]))
  h[ids]=np.clip(seed[ids]+(2*u-1)*span,hlo[ids]+.02,hhi[ids]-.02)
  kd.qpos[layout["hands"]["left"]["qadr"]]=h;mujoco.mj_forward(km,kd)
  gap,_=ref.distance(km,kd,tdg,ixg)
  err=abs(gap-(PHONE[1]+.0035))
  if PHONE[1]+.002<=gap<=PHONE[1]+.0055:
   candidates.append((err,h.copy(),gap))
 if not candidates:
  raise RuntimeError("PREGRASP_REACH_BLOCKED: no 2-5 mm open C-gap branch")
 # Validate PREGRASP against the actual finite-size phone, not merely against
 # the opposing fingertip.  This prevents an "open" finger link from already
 # intersecting a broad phone face.
 probe=wrapper(pc,.7);pd=mujoco.MjData(probe)
 for oj in range(km.njnt):
  n=mujoco.mj_id2name(km,mujoco.mjtObj.mjOBJ_JOINT,oj)
  nj=mujoco.mj_name2id(probe,mujoco.mjtObj.mjOBJ_JOINT,n)
  if nj<0: continue
  typ=km.jnt_type[oj];width=7 if typ==mujoco.mjtJoint.mjJNT_FREE else 4 if typ==mujoco.mjtJoint.mjJNT_BALL else 1
  oa=km.jnt_qposadr[oj];na=probe.jnt_qposadr[nj]
  pd.qpos[na:na+width]=km.key_qpos[0,oa:oa+width]
 def pq(name):
  j=mujoco.mj_name2id(probe,mujoco.mjtObj.mjOBJ_JOINT,name);return probe.jnt_qposadr[j]
 parmq=np.array([pq(mujoco.mj_id2name(km,mujoco.mjtObj.mjOBJ_JOINT,j)) for j in info["arm_joint_ids"]])
 phq=np.array([pq(n) for n in layout["hands"]["left"]["names"]])
 pd.qpos[parmq[:7]]=qarm;pd.qpos[parmq[7:]]=right;pd.qpos[phq]=seed
 pfj=mujoco.mj_name2id(probe,mujoco.mjtObj.mjOBJ_JOINT,"phone_free");pfa=probe.jnt_qposadr[pfj]
 pd.qpos[pfa:pfa+7]=np.r_[pc,1,0,0,0]
 pg=mujoco.mj_name2id(probe,mujoco.mjtObj.mjOBJ_GEOM,"physics_phone_geom")
 pt=ref.collision_geoms(probe,"left_hand_thumb_2_link")[-1]
 pi=ref.collision_geoms(probe,"left_hand_index_1_link")[-1]
 safe=[]
 # Coarse-to-fine temporary proxy placement around the reachable edge point.
 # The wrapper table will be rebuilt under the selected proxy pose.
 offsets=[np.array([dx,dy,dz]) for dx in (-.012,0,.012)
          for dy in (-.008,0,.008) for dz in np.linspace(-.008,.008,9)]
 for err,h,gap in sorted(candidates,key=lambda x:x[0])[:500]:
  for off in offsets:
   trial_pc=pc+off;pd.qpos[pfa:pfa+3]=trial_pc;pd.qpos[phq]=h;mujoco.mj_forward(probe,pd)
   ds=[float(ref.distance(probe,pd,x,pg)[0]) for x in (pt,pi)]
   forbidden=[]
   for geom in range(probe.ngeom):
    if geom in (pg,pt,pi): continue
    body=old.body_name(probe,geom)
    if body and (body.startswith("left_hand_") or body.startswith("right_hand_")):
     forbidden.append(float(ref.distance(probe,pd,geom,pg)[0]))
   if all(.002<=d<=.005 for d in ds) and min(forbidden,default=1.)>=0:
    pd.qpos[phq]=seed;mujoco.mj_forward(probe,pd)
    close_ds=[float(ref.distance(probe,pd,x,pg)[0]) for x in (pt,pi)]
    # A gradual trajectory may stop at first bilateral contact; require the
    # validated close branch to at least cross both phone surfaces.
    if max(close_ds)<=.001:
     score=sum(abs(d-.0035) for d in ds)+sum(abs(d) for d in close_ds)
     safe.append((score,h,gap,ds,min(forbidden,default=1.),trial_pc.copy(),close_ds))
 if not safe:
  raise RuntimeError("PREGRASP_REACH_BLOCKED: finite phone has no collision-free 2-5 mm two-sided clearance")
 _,pre,pre_gap,pre_phone_dist,pre_forbidden,pc,planned_close_dist=min(safe,key=lambda x:x[0])
 open_hand=natural["full_qpos"][layout["hands"]["left"]["qadr"]].copy()
 results=[];all_pose=[];all_force=[];all_track=[]
 grasp_q=None
 for mu in (.4,.7,1.0):
  model=wrapper(pc,mu);data=mujoco.MjData(model)
  # Adding a free phone before the robot changes qpos ordering.  Copy the
  # original keyframe joint-by-joint instead of reusing the reordered key.
  for oj in range(km.njnt):
   n=mujoco.mj_id2name(km,mujoco.mjtObj.mjOBJ_JOINT,oj)
   nj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
   if nj<0: continue
   typ=km.jnt_type[oj]
   width=7 if typ==mujoco.mjtJoint.mjJNT_FREE else 4 if typ==mujoco.mjtJoint.mjJNT_BALL else 1
   oa=km.jnt_qposadr[oj];na=model.jnt_qposadr[nj]
   data.qpos[na:na+width]=km.key_qpos[0,oa:oa+width]
  # Resolve qpos/actuator addresses in the independently compiled wrapper.
  # validate_model.joint_names contains every robot joint; use the explicit
  # 14 arm joint IDs so controls cannot be accidentally sent to hips/legs.
  names=[mujoco.mj_id2name(km,mujoco.mjtObj.mjOBJ_JOINT,j)
         for j in info["arm_joint_ids"]]
  jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
  qadr=np.array([model.jnt_qposadr[j] for j in jids]);hand_names=layout["hands"]["left"]["names"]+layout["hands"]["right"]["names"]
  hj=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in hand_names];hq=np.array([model.jnt_qposadr[j] for j in hj])
  data.qpos[qadr[:7]]=qarm;data.qpos[qadr[7:]]=right;data.qpos[hq[:7]]=pre;data.qpos[hq[7:]]=rhand
  # Set phone freejoint exactly horizontal at the reachable temporary proxy.
  fj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"phone_free");fq=model.jnt_qposadr[fj]
  data.qpos[fq:fq+7]=np.r_[pc,1,0,0,0];mujoco.mj_forward(model,data)
  phone_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"physics_phone_geom")
  thumb_geom=ref.collision_geoms(model,"left_hand_thumb_2_link")[-1]
  index_geom=ref.collision_geoms(model,"left_hand_index_1_link")[-1]
  pre_dist=[float(ref.distance(model,data,g,phone_geom)[0]) for g in (thumb_geom,index_geom)]
  saved=data.qpos[hq[:7]].copy();data.qpos[hq[:7]]=seed;mujoco.mj_forward(model,data)
  closed_dist=[float(ref.distance(model,data,g,phone_geom)[0]) for g in (thumb_geom,index_geom)]
  data.qpos[hq[:7]]=saved;mujoco.mj_forward(model,data)
  anames=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)]
  amap={n:i for i,n in enumerate(anames)}
  def controls(la,lh):
   for n,v in zip(names[:7],la):
    if n in amap:data.ctrl[amap[n]]=v
   for n,v in zip(names[7:],right):
    if n in amap:data.ctrl[amap[n]]=v
   for n,v in zip(layout["hands"]["left"]["names"],lh):
    if n in amap:data.ctrl[amap[n]]=v
   for n,v in zip(layout["hands"]["right"]["names"],rhand):
    if n in amap:data.ctrl[amap[n]]=v
  dt=model.opt.timestep;steps=int(14/dt);contact_step=None;contact_hold=0;maxpen=0;maxpen_pair=None;maxpen_time=None;forces=[];poses=[];tracks=[]
  thumb_body="left_hand_thumb_2_link";index_body="left_hand_index_1_link"
  liftq=qarm.copy()
  controls(qarm,pre)
  for k in range(steps):
   t=k*dt
   if t<1:target=pre
   elif t<7 and contact_step is None:target=pre+(t-1)/6*(seed-pre)
   else:target=(data.qpos[hq[:7]].copy() if contact_step is not None else seed)
   # Once two-sided contact forms, hold fingers and execute a Cartesian +10 cm
   # wrist lift obtained from bounded left-arm IK.
   lift_start=(contact_step*dt+.75) if contact_step is not None else 99.
   if contact_step is not None and t>lift_start:
    u=min(1,(t-lift_start)/3);liftq=qarm+u*(qlift-qarm)
   controls(liftq,target);mujoco.mj_step(model,data)
   th=False;ix=False;fth=0.;fix=0.
   for c in data.contact:
    a=old.body_name(model,c.geom1);b=old.body_name(model,c.geom2)
    if "physics_phone" not in (a,b):continue
    other=b if a=="physics_phone" else a;force=float(data.efc_force[c.efc_address]) if c.efc_address>=0 else 0
    # Table support penetration is not finger-phone penetration.
    if other and other.startswith(("left_hand_","right_hand_")):
     pen=max(0,-float(c.dist))
     if pen>maxpen:
      maxpen=pen;maxpen_time=t
      maxpen_pair=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),
                   mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),
                   a,b]
    if other==thumb_body:th=True;fth=max(fth,force)
    if other==index_body:ix=True;fix=max(fix,force)
   if th and ix:
    contact_hold+=1
    if contact_step is None:contact_step=k
   phonepos=data.qpos[fq:fq+3].copy();phonequat=data.qpos[fq+3:fq+7].copy()
   poses.append(np.r_[t,phonepos,phonequat]);forces.append([t,fth,fix,maxpen])
   tracks.append([t,*data.qpos[qadr[:7]],*liftq])
  poses=np.asarray(poses);forces=np.asarray(forces);tracks=np.asarray(tracks)
  lift=float(poses[:,3].max()-poses[0,3]);drop=bool(poses[-1,3]<poses[0,3]-.02)
  stable=bool(contact_step is not None and lift>=.05 and not drop and maxpen<.001)
  verdict=("PHONE_LIFT_STABLE_NO_PASSIVE_ROTATION" if stable else
    "PHONE_GRASP_EXCESSIVE_PENETRATION" if maxpen>=.001 else
    "PHONE_LIFT_DROP" if drop else "PHONE_LIFT_SLIP" if contact_step is not None else "CONTACT_NOT_FORMED")
  rec={"friction":mu,"verdict":verdict,"contact_formed":contact_step is not None,
   "contact_duration_s":contact_hold*dt,"max_penetration_m":maxpen,"phone_lift_height_m":lift,
   "phone_drop":drop,"max_thumb_force_n":float(forces[:,1].max()),"max_index_force_n":float(forces[:,2].max()),
   "max_penetration_pair":maxpen_pair,"max_penetration_time_s":maxpen_time,
   "pregrasp_thumb_index_distance_m":pre_dist,"closed_target_thumb_index_distance_m":closed_dist}
  results.append(rec);all_pose.append(poses);all_force.append(forces);all_track.append(tracks)
  if stable and grasp_q is None:grasp_q=data.qpos[hq[:7]].copy()
 with (OUT/"friction_comparison.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,results[0].keys());w.writeheader();w.writerows(results)
 np.savez_compressed(OUT/"phone_pose_trajectory.npz",frictions=np.array([.4,.7,1.]),trajectories=np.asarray(all_pose))
 with (OUT/"contact_force_trajectory.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["friction","time_s","thumb_force_n","index_force_n","max_penetration_m"])
  for mu,a in zip((.4,.7,1.),all_force):
   for r in a:w.writerow([mu,*r])
 with (OUT/"joint_tracking.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["friction","time_s",*["q"+str(i) for i in range(7)],*["target"+str(i) for i in range(7)]])
  for mu,a in zip((.4,.7,1.),all_track):
   for r in a:w.writerow([mu,*r])
 ready=grasp_q is not None
 report={"verdict":"G1_LEFT_PHONE_PHYSICS_GRASP_READY" if ready else "G1_LEFT_PHONE_PHYSICS_GRASP_BLOCKED",
  "friction_results":results,"pregrasp_qpos":pre,"close_target_qpos":seed,
  "pregrasp_fingertip_clearance_m":pre_phone_dist,
  "pregrasp_forbidden_clearance_m":pre_forbidden,
  "phone_proxy_position_m":pc,
  "arm_pregrasp_qpos":qarm,"arm_lift_10cm_ik_qpos":qlift,
  "failed_static_complete_qpos_used":False,
  "phone_welded":False,"phone_mocap":False,"isaac_lab_executed":False,
  "physics_videos_generated":False,
  "physics_video_gate_reason":None if ready else "no friction condition passed penetration and 5 cm lift gates",
  "predefined_primitive_saved":ready,"full_episode_trajectory_generated":False,
  "largest_blocker":None if ready else ("EXCESSIVE_PENETRATION" if any(r["max_penetration_m"]>=.001 for r in results) else "CONTACT_NOT_FORMED")}
 old.atomic_json(OUT/"physics_grasp_report.json",report)
 if ready:
  old.atomic_json(PRIM/"dex3_phone_open_pregrasp_grasp.json",{"OPEN":open_hand,"PREGRASP":pre,
   "PHYSICS_CONTACT_GRASP":grasp_q,"transition_rate_rad_s":.15,"friction_results":results})
 print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 0 if ready else 2
if __name__=="__main__":raise SystemExit(main())
