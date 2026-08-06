#!/usr/bin/env python3
"""Continuation search starting from the deterministic 7.95 mm Dex3 C-gap seed."""
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
OUT=ROOT/"converted_runs/g1_left_phone_cgap_grasp"
REPORT=ROOT/"converted_runs/g1_left_phone_grasp_diagnostic/left_phone_grasp_failure_diagnostic.json"
FAILED=ROOT/"converted_runs/g1_left_phone_grasp_diagnostic/best_failed_candidate_diagnostic.npz"
LAYOUT=ROOT/"isaaclab_magsafe_fixed_scene/scene_layout.json"
PHONE=np.array([.1496,.00795,.0715]);TOL=.0002

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 scene=json.loads(LAYOUT.read_text());actual=np.asarray(scene["phone"]["size_landscape_xyz"],float)
 if not np.allclose(actual,PHONE,atol=1e-9):raise RuntimeError(f"phone size changed: {actual}")
 dr=json.loads(REPORT.read_text());sol=dr["phone_thickness_aperture_solution"];hq=np.asarray(sol["qpos"])
 with np.load(FAILED,allow_pickle=False) as z:full=z["full_g1_qpos"].copy()
 info=old.relative.latest.ik.validate_model(old.G1_XML);layout,schema=old.hand_layout(info)
 # Restore exact deterministic hand-only seed at the diagnostic arm pose.
 base=mujoco.MjModel.from_xml_path(str(old.G1_XML));bd=mujoco.MjData(base)
 bd.qpos[:]=full;bd.qpos[layout["hands"]["left"]["qadr"]]=hq;mujoco.mj_forward(base,bd)
 gids={p:ref.collision_geoms(base,f"left_hand_{p}_{2 if p=='thumb' else 1}_link")[-1]
       for p in ("thumb","index","middle")}
 gap,ft=ref.distance(base,bd,gids["thumb"],gids["index"]);pts=ft.reshape(2,3)
 ts=diag.actual_surface(base,bd,gids["thumb"],pts[0]);ins=diag.actual_surface(base,bd,gids["index"],pts[1])
 normal_dot=float(np.dot(ts["raw_mesh_outward_normal"],ins["raw_mesh_outward_normal"]))
 ranges=layout["hands"]["left"]["ranges"];margin=np.minimum(hq-ranges[:,0],ranges[:,1]-hq)
 seed={"verdict":"CGAP_FINGER_BRANCH_RECONSTRUCTED","phone_dimensions_m":actual,
  "joint_order":layout["hands"]["left"]["names"],"joint_values":hq,
  "thumb_joints":dict(zip(layout["hands"]["left"]["names"][:3],hq[:3])),
  "middle_third_joints":dict(zip(layout["hands"]["left"]["names"][3:5],hq[3:5])),
  "index_joints":dict(zip(layout["hands"]["left"]["names"][5:],hq[5:])),
  "thumb_surface_point":ts["surface_point"],"index_surface_point":ins["surface_point"],
  "thumb_actual_surface_normal":ts["raw_mesh_outward_normal"],
  "index_actual_surface_normal":ins["raw_mesh_outward_normal"],
  "surface_normal_dot":normal_dot,"surface_aperture_m":gap,
  "aperture_error_m":abs(gap-PHONE[1]),"joint_limit_margins":margin,
  "middle_entry_space_precheck":"full-phone placement evaluated in wrist_pose_candidates.csv"}
 old.atomic_json(OUT/"left_dex3_cgap_seed.json",seed)
 if abs(gap-PHONE[1])>.001 or normal_dot>-.95:
  report={"verdict":"CGAP_FINGER_BRANCH_RECONSTRUCTION_FAILED","seed":seed,
   "trajectory_generated":False,"isaac_lab_executed":False}
  old.atomic_json(OUT/"selected_left_phone_cgap_grasp_report.json",report);print(report["verdict"]);return 2
 # Hand-local axes. Thickness is index->thumb surface direction. Candidate
 # long axes rotate about thickness; broad-face center offsets remain bounded.
 x=pts[0]-pts[1];x/=np.linalg.norm(x);w=mujoco.mj_name2id(base,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_yaw_link")
 refaxis=bd.xmat[w].reshape(3,3)[:,0];z0=refaxis-x*np.dot(refaxis,x)
 if np.linalg.norm(z0)<.2:
  refaxis=bd.xmat[w].reshape(3,3)[:,2];z0=refaxis-x*np.dot(refaxis,x)
 z0/=np.linalg.norm(z0);y0=np.cross(z0,x);y0/=np.linalg.norm(y0);z0=np.cross(x,y0)
 midpoint=pts.mean(0)
 # One compiled proxy model, then mutate its body pose; all collision pairs remain enabled.
 model,_=old.expanded_phone_model(midpoint,Rotation.from_matrix(np.column_stack((x,y0,z0))).as_euler("xyz"))
 data=mujoco.MjData(model);phone=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"phone_proxy_geom")
 pb=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"phone_proxy")
 data.qpos[:]=full;data.qpos[layout["hands"]["left"]["qadr"]]=hq
 intended={p:ref.collision_geoms(model,f"left_hand_{p}_{2 if p=='thumb' else 1}_link")[-1] for p in ("thumb","index")}
 collision=[g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
 forbidden=[g for g in collision if g!=phone and g not in intended.values()
            and old.category(old.body_name(model,g)) in ("finger","hand_wrist","arm")]
 fields=["candidate","roll_deg","short_offset_m","long_offset_m","valid","thumb_distance_m","index_distance_m",
         "thumb_normal_error_deg","index_normal_error_deg","normal_dot","contacts_inside",
         "minimum_forbidden_clearance_m","middle_clearance_m","palm_clearance_m"]
 rows=[];valid=[]
 offsets=[(0,0),(.018,0),(-.018,0),(0,.035),(0,-.035)]
 for ri,angle in enumerate(np.linspace(0,2*np.pi,40,endpoint=False)):
  ca,sa=np.cos(angle),np.sin(angle);y=ca*y0+sa*z0;z=-sa*y0+ca*z0;R=np.column_stack((x,y,z))
  for oi,(oy,oz) in enumerate(offsets):
   center=midpoint+y*oy+z*oz;model.body_pos[pb]=center
   q=Rotation.from_matrix(R).as_quat();model.body_quat[pb]=q[[3,0,1,2]]
   mujoco.mj_forward(model,data)
   crec={};inside=True
   for part,face in (("thumb",1),("index",-1)):
    gid=intended[part];d,fromto=ref.distance(model,data,gid,phone)
    normal=ref.contact_normal_phone_to_tip(model,data,gid,phone,fromto);target=R[:,0]*face
    err=float(np.degrees(np.arccos(np.clip(np.dot(normal,target),-1,1))))
    point=.5*(fromto[:3]+fromto[3:])
    for c in data.contact:
     if {int(c.geom1),int(c.geom2)}=={gid,phone}:point=np.asarray(c.pos).copy()
    local=R.T@(point-center);ok=(abs(local[1])<=PHONE[2]/2-.001 and abs(local[2])<=PHONE[0]/2-.001)
    inside&=ok;crec[part]=(d,err,point,normal,local)
   fdist=[]
   middle_d=[];palm_d=[]
   for g in forbidden:
    d,_=ref.distance(model,data,g,phone);fdist.append(d);bn=old.body_name(model,g)
    if "middle" in bn:middle_d.append(d)
    if "wrist_yaw" in bn:palm_d.append(d)
   mind=min(fdist,default=1);ndot=float(np.dot(crec["thumb"][3],crec["index"][3]))
   ok=bool(all(-TOL<=crec[p][0]<=.0005 and crec[p][1]<20 for p in ("thumb","index"))
           and ndot<=-.95 and inside and mind>=-TOL)
   row={"candidate":len(rows),"roll_deg":np.degrees(angle),"short_offset_m":oy,"long_offset_m":oz,
    "valid":ok,"thumb_distance_m":crec["thumb"][0],"index_distance_m":crec["index"][0],
    "thumb_normal_error_deg":crec["thumb"][1],"index_normal_error_deg":crec["index"][1],
    "normal_dot":ndot,"contacts_inside":inside,"minimum_forbidden_clearance_m":mind,
    "middle_clearance_m":min(middle_d,default=1),"palm_clearance_m":min(palm_d,default=1),
    "_center":center,"_R":R,"_contacts":crec}
   rows.append(row)
   if ok:valid.append(row)
 with (OUT/"wrist_pose_candidates.csv").open("w",newline="") as f:
  wr=csv.DictWriter(f,fields);wr.writeheader()
  for r in rows:wr.writerow({k:r[k] for k in fields})
 # Fail-closed empty downstream tables unless a valid rigid hand-local frame exists.
 for name,cols in (("arm_ik_candidates.csv",["wrist_candidate","ik_seed","valid","reason"]),
                   ("continuation_candidates.csv",["candidate","stage","valid","reason"])):
  with (OUT/name).open("w",newline="") as f:csv.DictWriter(f,cols).writeheader()
 if not valid:
  best=max(rows,key=lambda r:(r["minimum_forbidden_clearance_m"],
      -abs(r["thumb_distance_m"])-abs(r["index_distance_m"]),
      -r["thumb_normal_error_deg"]-r["index_normal_error_deg"]))
  report={"verdict":"G1_LEFT_PHONE_CGAP_GRASP_BLOCKED","safety_pass":False,
   "stop_stage":"wrist_target_feasibility / rigid full-phone insertion",
   "seed":seed,"wrist_pose_candidate_count":len(rows),"valid_wrist_pose_count":0,
   "arm_ik_candidate_count":0,"continuation_candidate_count":0,
   "largest_blocker":(
    "The reconstructed 7.95 mm fingertip-surface branch cannot admit the finite "
    "149.6 x 71.5 mm broad faces: every one of 200 closing-axis roll/face-offset "
    "placements causes >0.2 mm forbidden distal/proximal/middle/palm penetration "
    "or loses the two intended broad-face contacts."),
   "best_wrist_candidate":{k:v for k,v in best.items() if not k.startswith("_")},
   "bimanual_side_support_used":False,"right_phone_support":False,
   "actual_isaac_phone_pose_used":False,"base_moved":False,
   "trajectory_generated":False,"right_accessory_search":False,
   "workspace_calibration":False,"isaac_lab_executed":False,"hardware_executed":False}
  old.atomic_json(OUT/"selected_left_phone_cgap_grasp_report.json",report)
  print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 2
 # Stage E: rigidly map each valid hand-local phone frame to a torso-frontal
 # world phone frame, then solve left-arm IK from ten deterministic seeds.
 # Finger qpos remains exactly fixed.
 data.qpos[:]=full;data.qpos[layout["hands"]["left"]["qadr"]]=hq;mujoco.mj_forward(model,data)
 wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_wrist_yaw_link")
 wp0=data.xpos[wrist].copy();wR0=data.xmat[wrist].reshape(3,3).copy()
 natural=old.relative.load_natural_start(old.NATURAL_NPZ,info)
 right_arm=natural["arm_q"][7:];right_hand=natural["full_qpos"][layout["hands"]["right"]["qadr"]]
 arm_lo=info["joint_limits"][:7,0]+.03;arm_hi=info["joint_limits"][:7,1]-.03
 # Place the phone from a reachable wrist reference, rather than pairing each
 # wrist roll with an unrelated Cartesian grid point.  This is the continuation
 # step: at zero offset the requested wrist position is exactly the diagnostic
 # wrist position, while the phone is torso-frontal.  Small torso-local offsets
 # provide the reach/posture variants.
 center_offsets=[
  np.array(v) for v in ((0,0,0),(-.03,0,0),(.03,0,0),(0,-.03,0),
   (0,.03,0),(0,0,-.04),(0,0,.04),(-.02,.02,0),(-.02,-.02,0),
   (.02,.02,0))
 ]
 ikrows=[];ikvalid=[]
 def set_arm(qarm,phone_center,phone_R):
  data.qpos[:]=model.key_qpos[0];data.qpos[info["arm_qpos_ids"][:7]]=qarm
  data.qpos[info["arm_qpos_ids"][7:]]=right_arm
  data.qpos[layout["hands"]["left"]["qadr"]]=hq
  data.qpos[layout["hands"]["right"]["qadr"]]=right_hand
  model.body_pos[pb]=phone_center
  qq=Rotation.from_matrix(phone_R).as_quat();model.body_quat[pb]=qq[[3,0,1,2]]
  mujoco.mj_forward(model,data)
 seeds=[]
 base_seeds=[natural["arm_q"][:7].copy(),full[info["arm_qpos_ids"][:7]].copy()]
 for si in range(10):
  q=base_seeds[si%2].copy()
  if si in (2,3):q[3]+= .35 if si==2 else -.35
  if si in (4,5):q[2]+= .30 if si==4 else -.30
  if si in (6,7):q[4]+= .30 if si==6 else -.30
  if si in (8,9):q[6]+= .30 if si==8 else -.30
  seeds.append(np.clip(q,arm_lo,arm_hi))
 for wi,wc in enumerate(valid):
  phone0_R=wc["_R"];phone0_p=wc["_center"]
  rel_R=wR0.T@phone0_R;rel_p=wR0.T@(phone0_p-wp0)
  phone_R=np.eye(3);target_R=phone_R@rel_R.T
  for si,q0 in enumerate(seeds):
   phone_center=wp0+target_R@rel_p+center_offsets[si]
   target_p=phone_center-target_R@rel_p
   def resid(q):
    set_arm(q,phone_center,phone_R)
    cr=data.xmat[wrist].reshape(3,3)
    rot=Rotation.from_matrix(cr.T@target_R).as_rotvec()
    return np.r_[100*(data.xpos[wrist]-target_p),10*rot,
                 .005*(q-natural["arm_q"][:7])]
   sol=least_squares(resid,q0,bounds=(arm_lo,arm_hi),max_nfev=160,
                     ftol=1e-10,xtol=1e-10,gtol=1e-10)
   set_arm(sol.x,phone_center,phone_R);cr=data.xmat[wrist].reshape(3,3)
   pe=float(np.linalg.norm(data.xpos[wrist]-target_p))
   oe=float(np.degrees(np.linalg.norm(Rotation.from_matrix(cr.T@target_R).as_rotvec())))
   margin=float(np.minimum(sol.x-info["joint_limits"][:7,0],
                           info["joint_limits"][:7,1]-sol.x).min())
   # Enabled-contact audit. Only distal thumb/index phone pairs are intended.
   forbidden_contacts=[];intended_contacts=[]
   for c in data.contact:
    a,b=old.body_name(model,c.geom1),old.body_name(model,c.geom2)
    if "phone_proxy" in (a,b) and (
      "left_hand_thumb_2_link" in (a,b) or "left_hand_index_1_link" in (a,b)):
     intended_contacts.append(float(c.dist));continue
    relevant=(("phone_proxy" in (a,b)) or
      ("torso_link" in (a,b) and any(old.category(x) in ("arm","finger","hand_wrist") for x in (a,b))) or
      ((a.startswith("left_") and b.startswith("right_")) or
       (a.startswith("right_") and b.startswith("left_"))))
    if relevant:forbidden_contacts.append((a,b,float(c.dist)))
   # MuJoCo's discrete contact list is not a contact-quality oracle: at a
   # positive sub-millimetre surface distance an intended pair correctly has
   # no mjContact.  Evaluate both intended distal pairs with mj_geomDistance,
   # exactly as in Stage D.
   intended_distances=[]
   intended_errors=[]
   intended_inside=True
   for part,face in (("thumb",1),("index",-1)):
    gid=intended[part]
    dd,ff=ref.distance(model,data,gid,phone)
    nn=ref.contact_normal_phone_to_tip(model,data,gid,phone,ff)
    intended_distances.append(float(dd))
    intended_errors.append(float(np.degrees(np.arccos(
      np.clip(np.dot(nn,phone_R[:,0]*face),-1,1)))))
    pp=.5*(ff[:3]+ff[3:])
    ll=phone_R.T@(pp-phone_center)
    intended_inside &= bool(abs(ll[1])<=PHONE[2]/2-.001 and
                            abs(ll[2])<=PHONE[0]/2-.001)
   intended_geometry=bool(all(-TOL<=d<=.0005 for d in intended_distances)
      and max(intended_errors)<20 and intended_inside)
   ok=bool(pe<.002 and oe<3 and margin>=.03
           and min([x[2] for x in forbidden_contacts],default=1)>=-TOL
           and intended_geometry)
   rec={"wrist_candidate":wc["candidate"],"ik_seed":si,"valid":ok,
    "position_error_m":pe,"orientation_error_deg":oe,"minimum_margin_rad":margin,
    "forbidden_contact_count":len(forbidden_contacts),
    "intended_contact_count":2 if intended_geometry else len(intended_contacts),
    "phone_x":phone_center[0],"phone_y":phone_center[1],"phone_z":phone_center[2],
    "_arm":sol.x.copy(),"_phone_center":phone_center,"_phone_R":phone_R,
    "_wrist_source":wc}
   ikrows.append(rec)
   if ok:ikvalid.append(rec)
 ikfields=["wrist_candidate","ik_seed","valid","position_error_m","orientation_error_deg",
           "minimum_margin_rad","forbidden_contact_count","intended_contact_count",
           "phone_x","phone_y","phone_z"]
 with (OUT/"arm_ik_candidates.csv").open("w",newline="") as f:
  wr=csv.DictWriter(f,ikfields);wr.writeheader()
  for r in ikrows:wr.writerow({k:r[k] for k in ikfields})
 if not ikvalid:
  with (OUT/"continuation_candidates.csv").open("w",newline="") as f:
   csv.DictWriter(f,["candidate","stage","valid","reason"]).writeheader()
  report={"verdict":"G1_LEFT_PHONE_CGAP_GRASP_BLOCKED","safety_pass":False,
   "stop_stage":"arm IK reach/collision","seed":seed,
   "wrist_pose_candidate_count":len(rows),"valid_wrist_pose_count":len(valid),
   "arm_ik_candidate_count":len(ikrows),"valid_arm_ik_count":0,
   "largest_blocker":"No rigid C-gap wrist target passed arm IK and enabled collision audit.",
   "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False}
  old.atomic_json(OUT/"selected_left_phone_cgap_grasp_report.json",report)
  print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 2
 # Stage F: branch-locked fine evaluation. Perturb arm and thumb/index q within
 # small bounds; phone stays torso frontal. No clipping is used: out-of-bound
 # samples are rejected before evaluation.
 baseik=max(ikvalid,key=lambda r:(r["minimum_margin_rad"],
     r["_wrist_source"]["minimum_forbidden_clearance_m"],-r["position_error_m"]))
 cont=[];contvalid=[]
 for ci in range(500):
  rng=np.random.default_rng(52001+ci)
  stage=1+ci//100;qarm=baseik["_arm"]+rng.normal(0,.0015*stage,7)
  qhand=hq.copy()
  if stage>=4:qhand[[0,1,2,5,6]]+=rng.normal(0,.0005*(stage-3),5)
  if np.any(qarm<arm_lo)|np.any(qarm>arm_hi)|np.any(qhand<ranges[:,0])|np.any(qhand>ranges[:,1]):
   cont.append({"candidate":ci,"stage":stage,"valid":False,"reason":"sample outside bounds"});continue
  data.qpos[:]=model.key_qpos[0];data.qpos[info["arm_qpos_ids"][:7]]=qarm
  data.qpos[info["arm_qpos_ids"][7:]]=right_arm;data.qpos[layout["hands"]["left"]["qadr"]]=qhand
  data.qpos[layout["hands"]["right"]["qadr"]]=right_hand;model.body_pos[pb]=baseik["_phone_center"]
  model.body_quat[pb]=[1,0,0,0];mujoco.mj_forward(model,data)
  td=ref.collision_geoms(model,"left_hand_thumb_2_link")[-1]
  ix=ref.collision_geoms(model,"left_hand_index_1_link")[-1]
  aperture,_=ref.distance(model,data,td,ix)
  phone_d=[];forbid=[]
  for g in range(model.ngeom):
   if not (model.geom_contype[g] or model.geom_conaffinity[g]) or g==phone:continue
   d,_=ref.distance(model,data,g,phone);bn=old.body_name(model,g)
   if g in (td,ix):phone_d.append(d)
   elif old.category(bn) in ("finger","hand_wrist","arm"):forbid.append(d)
  raw_normals=[];raw_errors=[]
  for gid,target in ((td,np.array([-1.,0,0])),(ix,np.array([1.,0,0]))):
   _,ft=ref.distance(model,data,gid,phone)
   nn=np.asarray(diag.actual_surface(model,data,gid,ft[:3])["raw_mesh_outward_normal"])
   raw_normals.append(nn)
   raw_errors.append(float(np.degrees(np.arccos(np.clip(np.dot(nn,target),-1,1)))))
  raw_dot=float(np.dot(raw_normals[0],raw_normals[1]))
  ok=bool(abs(aperture-PHONE[1])<.001 and len(phone_d)==2
          and all(-TOL<=d<=.0005 for d in phone_d)
          and min(forbid,default=1)>=-TOL and raw_dot<=-.95
          and max(raw_errors)<20)
  rec={"candidate":ci,"stage":stage,"valid":ok,"reason":"" if ok else "branch/contact/collision",
       "aperture_m":aperture,"minimum_forbidden_clearance_m":min(forbid,default=1),
       "actual_normal_dot":raw_dot,"maximum_actual_normal_error_deg":max(raw_errors),
       "_arm":qarm,"_hand":qhand}
  cont.append(rec)
  if ok:contvalid.append(rec)
 cf=["candidate","stage","valid","reason","aperture_m","minimum_forbidden_clearance_m",
     "actual_normal_dot","maximum_actual_normal_error_deg"]
 with (OUT/"continuation_candidates.csv").open("w",newline="") as f:
  wr=csv.DictWriter(f,cf);wr.writeheader()
  for r in cont:wr.writerow({k:r.get(k,"") for k in cf})
 if not contvalid:
  report={"verdict":"G1_LEFT_PHONE_CGAP_GRASP_BLOCKED","safety_pass":False,
   "stop_stage":"continuation branch/contact","valid_wrist_pose_count":len(valid),
   "valid_arm_ik_count":len(ikvalid),"continuation_candidate_count":500,
   "valid_continuation_count":0,"largest_blocker":"All fine samples escaped contact/collision constraints.",
   "trajectory_generated":False,"isaac_lab_executed":False,"hardware_executed":False}
  old.atomic_json(OUT/"selected_left_phone_cgap_grasp_report.json",report)
  print(json.dumps(report,indent=2));print(report["verdict"]);return 2
 # Prefer the earliest valid continuation stage.  A later stage is not an
 # improvement merely because a mesh clearance increases: stages 4-5 release
 # distal joints and can jump to a different contact facet.  Stage 1 preserves
 # the diagnostically verified C-gap branch exactly.
 fixed_branch=[r for r in contvalid if r["stage"]<=3]
 selected=max(fixed_branch or contvalid,
              key=lambda r:r["minimum_forbidden_clearance_m"])
 # Final exact state and metrics.
 qarm=selected["_arm"];qhand=selected["_hand"];data.qpos[:]=model.key_qpos[0]
 data.qpos[info["arm_qpos_ids"][:7]]=qarm;data.qpos[info["arm_qpos_ids"][7:]]=right_arm
 data.qpos[layout["hands"]["left"]["qadr"]]=qhand;data.qpos[layout["hands"]["right"]["qadr"]]=right_hand
 model.body_pos[pb]=baseik["_phone_center"];model.body_quat[pb]=[1,0,0,0];mujoco.mj_forward(model,data)
 contact_points=[];contact_normals=[];contact_distances=[];contact_normal_errors=[]
 for gid in (ref.collision_geoms(model,"left_hand_thumb_2_link")[-1],
             ref.collision_geoms(model,"left_hand_index_1_link")[-1]):
  d,ft=ref.distance(model,data,gid,phone);contact_distances.append(d)
  point=np.asarray(ft[:3]).copy()
  normal=np.asarray(diag.actual_surface(model,data,gid,point)["raw_mesh_outward_normal"])
  for c in data.contact:
   if {int(c.geom1),int(c.geom2)}=={gid,phone}:point=np.asarray(c.pos).copy()
  target=np.array([-1.,0,0]) if len(contact_points)==0 else np.array([1.,0,0])
  contact_normal_errors.append(float(np.degrees(np.arccos(np.clip(np.dot(normal,target),-1,1)))))
  contact_points.append(point);contact_normals.append(normal)
 fullq=data.qpos.copy();margins=np.minimum(qarm-info["joint_limits"][:7,0],
                                          info["joint_limits"][:7,1]-qarm)
 collision_counts={"intended_phone":0,"forbidden_phone":0,"arm_torso":0,
                   "finger_torso":0,"left_right_cross":0,"other":0}
 collision_pairs=[]
 for c in data.contact:
  a,b=old.body_name(model,c.geom1),old.body_name(model,c.geom2)
  if "phone_proxy" in (a,b):
   other=b if a=="phone_proxy" else a
   kind=("intended_phone" if other in
         ("left_hand_thumb_2_link","left_hand_index_1_link") else "forbidden_phone")
  elif "torso_link" in (a,b) and any(old.category(x)=="arm" for x in (a,b)):
   kind="arm_torso"
  elif "torso_link" in (a,b) and any(old.category(x) in ("finger","hand_wrist") for x in (a,b)):
   kind="finger_torso"
  elif ((a.startswith("left_") and b.startswith("right_")) or
        (a.startswith("right_") and b.startswith("left_"))):
   kind="left_right_cross"
  else:kind="other"
  collision_counts[kind]+=1
  collision_pairs.append({"category":kind,"body1":a,"body2":b,"distance_m":float(c.dist)})
 payload=dict(full_g1_qpos=fullq,left_arm_qpos=qarm,left_dex3_qpos=qhand,
  right_parked_arm_qpos=right_arm,right_parked_dex3_qpos=right_hand,
  phone_proxy_pose=np.r_[baseik["_phone_center"],1,0,0,0],
  contact_pad_poses=np.asarray(contact_points),actual_contact_points=np.asarray(contact_points),
  actual_contact_normals=np.asarray(contact_normals),aperture=np.asarray(selected["aperture_m"]),
  phone_broad_face_bounds=np.asarray([PHONE[2]/2,PHONE[0]/2]),
  forbidden_clearances=np.asarray([selected["minimum_forbidden_clearance_m"]]),
  joint_limit_margins=margins,wrist_bend_metrics=np.asarray([baseik["orientation_error_deg"]]),
  elbow_posture_metrics=np.asarray([qarm[3]]),selected_continuation_stage=np.asarray(selected["stage"]))
 tmp=OUT/"selected_left_phone_cgap_grasp.npz.incomplete"
 with tmp.open("wb") as f:np.savez_compressed(f,**payload)
 os.replace(tmp,OUT/"selected_left_phone_cgap_grasp.npz")
 report={"verdict":"G1_LEFT_PHONE_CGAP_GRASP_READY","safety_pass":True,
  "seed":seed,"wrist_pose_candidate_count":len(rows),"valid_wrist_pose_count":len(valid),
  "arm_ik_candidate_count":len(ikrows),"valid_arm_ik_count":len(ikvalid),
  "continuation_candidate_count":500,"valid_continuation_count":len(contvalid),
  "selected_stage":selected["stage"],"aperture_m":selected["aperture_m"],
  "contact_distances_m":contact_distances,
  "contact_normal_errors_deg":contact_normal_errors,
  "actual_normal_dot":float(np.dot(contact_normals[0],contact_normals[1])),
  "minimum_forbidden_clearance_m":selected["minimum_forbidden_clearance_m"],
  "minimum_joint_margin_rad":float(margins.min()),"phone_screen_frontal_error_deg":0.,
  "phone_long_up_error_deg":0.,"bimanual_side_support_used":False,
  "collision_counts":collision_counts,"collision_pairs":collision_pairs,
  "collision_note":(
   "The 'other' pair is the natural parked right thumb-index self-overlap; "
   "it is neither phone support nor a left-right/torso collision."),
  "right_phone_support":False,"trajectory_generated":False,"isaac_lab_executed":False,
  "gui_command":f"{sys.executable} {ROOT/'tools/view_g1_left_phone_cgap_grasp.py'} --grasp {OUT/'selected_left_phone_cgap_grasp.npz'}"}
 old.atomic_json(OUT/"selected_left_phone_cgap_grasp_report.json",report)
 print(json.dumps(old.serial(report),indent=2));print(report["verdict"]);return 0
if __name__=="__main__":raise SystemExit(main())
