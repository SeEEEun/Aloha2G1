#!/usr/bin/env python3
"""Right-middle-finger MagSafe ring insertion feasibility (simulation only)."""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/right_c_ring_mpl")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path('/home/jbnu/aloha_g1_dataset'); OUT=ROOT/'outputs/right_c_ring_insertion'
XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
SOURCE=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
NATURAL=ROOT/'converted_runs/magsafe_20260723_162750/dynamic_bimanual_spacing/g1_dynamic_bimanual_full_trajectory.npz'
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
import retarget_episode49_optimized_action_to_g1 as epi
import retarget_episode49_consensus_relative_bimanual_to_g1 as rel
import retarget_episode49_relative_bimanual_neutral_pinch_to_g1 as neutral

PHONE_CENTER=np.array([.525,.07,.83075]); PHONE_SIZE=np.array([.0715,.00795,.1496])
RING_CENTER=np.array([.525,.076425,.83075]); INNER=.0225; OUTER=.0275; DEPTH=.0035; GAP=36.
ROOT_POS=np.array([.44514890950197095,-.35257022755443246,.7922728583])
STATES=('preinsert','aperture_entry','inserted','hooked','removed')

def serial(x):
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,Path): return str(x)
 if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [serial(v) for v in x]
 return x
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(serial(x),indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def name(model,obj,i): return mujoco.mj_id2name(model,obj,int(i)) or f'id{i}'

def expanded_model():
 text=XML.read_text(); assets=XML.parent/'assets'
 text=text.replace('meshdir="assets/"',f'meshdir="{assets}/"').replace('meshdir="assets"',f'meshdir="{assets}"')
 geoms=[f'<geom name="phone_collision" type="box" pos="{PHONE_CENTER[0]} {PHONE_CENTER[1]} {PHONE_CENTER[2]}" size="{PHONE_SIZE[0]/2} {PHONE_SIZE[1]/2} {PHONE_SIZE[2]/2}" rgba=".2 .4 .9 .25" contype="1" conaffinity="1"/>']
 span=(360-GAP)/12; start=-90+GAP/2; rc=(INNER+OUTER)/2; rw=OUTER-INNER
 tang=2*rc*math.sin(math.radians(span)/2)*1.06
 for i in range(12):
  deg=start+(i+.5)*span; a=math.radians(deg); p=RING_CENTER+np.array([rc*math.cos(a),0,rc*math.sin(a)])
  geoms.append(f'<geom name="ring_segment_{i:02d}" type="box" pos="{p[0]} {p[1]} {p[2]}" euler="0 {a} 0" size="{tang/2} {DEPTH/2} {rw/2}" rgba=".05 .05 .05 .8" contype="1" conaffinity="1"/>')
 table='<geom name="table_collision" type="box" pos=".4175 .36 .7725" size=".4175 .36 .0225" rgba=".5 .5 .5 .2" contype="1" conaffinity="1"/>'
 text=text.replace('<worldbody>','<worldbody>\n'+table+'\n'+'\n'.join(geoms),1)
 td=tempfile.TemporaryDirectory(prefix='right_c_ring_'); p=Path(td.name)/'model.xml';p.write_text(text)
 return mujoco.MjModel.from_xml_path(str(p)),td

def context():
 info=epi.ik.validate_model(XML); start=rel.load_natural_start(NATURAL,info)
 model,tmp=expanded_model(); layout,_=neutral.hand_joint_schema(info)
 # qpos addresses are invariant after adding fixed world geoms.
 rarm=np.asarray(info['arm_qpos_ids'][7:],int); rq=np.asarray(layout['hands']['right']['qadr'],int)
 q0=start['full_qpos'].copy();q0[:3]=ROOT_POS
 # Keep the already registered root orientation from the active natural pose.
 q0[rarm]=start['arm_q'][7:]; return info,start,model,tmp,rarm,rq,q0,layout

def pose(model,data,b):
 i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,b);return data.xpos[i].copy(),data.xmat[i].reshape(3,3).copy()
def geom_pose(model,data,g):
 i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,g);return data.geom_xpos[i].copy(),data.geom_xmat[i].reshape(3,3).copy(),i

def audit_geometry(model,q0,rq):
 d=mujoco.MjData(model);d.qpos[:]=q0;mujoco.mj_forward(model,d)
 p,R,gid=geom_pose(model,d,'right_hand_middle_1_link') if False else geom_pose(model,d,name(model,mujoco.mjtObj.mjOBJ_GEOM,95))
 # Resolve by distal body, never by a presumed model-wide geom id.
 bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'right_hand_middle_1_link')
 gids=[i for i in range(model.ngeom) if model.geom_bodyid[i]==bid and (model.geom_contype[i] or model.geom_conaffinity[i])]
 gid=gids[-1];gname=name(model,mujoco.mjtObj.mjOBJ_GEOM,gid);p=data_p=d.geom_xpos[gid].copy();R=d.geom_xmat[gid].reshape(3,3).copy()
 tips=json.loads((ROOT/'configs/dex3_fingertip_frames.sim.json').read_text())['fingertips']['right_C']
 rec={'status':'PASS','simulation_only':True,'model':str(XML),'model_sha256':sha(XML),
 'scene_sources':{'scene_layout_sha256':sha(ROOT/'isaaclab_magsafe_fixed_scene/scene_layout.json'),'magsafe_scene_builder_sha256':sha(ROOT/'isaaclab_magsafe_fixed_scene/magsafe_scene_builder.py'),'generated_magsafe_fixed_scene_usda_sha256':sha(ROOT/'isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda')},'root_position':ROOT_POS,
 'portrait_phone':{'center':PHONE_CENTER,'size_world_xyz':PHONE_SIZE,'long_axis':[0,0,1],'back_normal':[0,1,0],'basis':'approved initial center; semantic portrait axes'},
 'right_C':{'joint_chain':['right_hand_middle_0_joint','right_hand_middle_1_joint'],'distal_link':'right_hand_middle_1_link','collision_geom_id':gid,'collision_geom_name':gname,'contact_pad_center_local':tips['local_position_xyz_m'],'pad_normal_local':tips['local_normal'],'flexion_joint_axis':tips['flexion_direction_joint_axis'],'pad_half_extents_m':tips['pad_half_extent_m']},
 'ring':{'center':RING_CENTER,'plane':'world XZ','normal':[0,1,0],'inner_radius_m':INNER,'outer_radius_m':OUTER,'depth_m':DEPTH,'gap_degrees':GAP,'collision':'12 convex box segments copied from active scene builder'},
 'hard_radial_clearance_m':INNER-max(sorted(tips['pad_half_extent_m'])[1:])}
 dump(OUT/'active_geometry_audit.json',rec);dump(OUT/'right_c_tip_geometry.json',rec['right_C']);dump(OUT/'ring_geometry.json',rec['ring']);return rec,gid

def desired_R(radial,normal=np.array([0.,-1.,0.])):
 # distal local +X points through aperture; local +Z follows radial direction.
 x=normal/np.linalg.norm(normal);z=radial-x*np.dot(radial,x);z/=np.linalg.norm(z);y=np.cross(z,x)
 return np.column_stack((x,y,z))

def solve_pose(model,qbase,rarm,rq,target,Rtar,hand_seed,max_nfev=100):
 d=mujoco.MjData(model); idx=np.r_[rarm,rq]; lo=model.jnt_range[model.dof_jntid[np.r_[model.jnt_dofadr[[model.jnt_qposadr.tolist().index(int(i)) if False else 0 for i in []]]]]] if False else np.zeros(0)
 lower=[];upper=[]
 for adr in idx:
  jid=int(np.where(model.jnt_qposadr==adr)[0][0]);a,b=model.jnt_range[jid];lower.append(a+1e-4);upper.append(b-1e-4)
 x0=np.clip(np.r_[qbase[rarm],hand_seed],lower,upper)
 def fun(x):
  d.qpos[:]=qbase;d.qpos[idx]=x;mujoco.mj_forward(model,d);p,R,_=geom_pose(model,d,name(model,mujoco.mjtObj.mjOBJ_GEOM,95)) if False else (None,None,None)
  bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'right_hand_middle_1_link'); gids=[i for i in range(model.ngeom) if model.geom_bodyid[i]==bid and (model.geom_contype[i] or model.geom_conaffinity[i])];p=d.geom_xpos[gids[-1]];R=d.geom_xmat[gids[-1]].reshape(3,3)
  rv=Rotation.from_matrix(R.T@Rtar).as_rotvec();return np.r_[60*(p-target),2.2*rv,.04*(x[:7]-qbase[rarm]),.015*(x[7:]-hand_seed)]
 sol=least_squares(fun,x0,bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 q=qbase.copy();q[idx]=sol.x;mujoco.mj_forward(model,d);return q,sol

def contacts(model,q,allow_hook=False):
 d=mujoco.MjData(model);d.qpos[:]=q;mujoco.mj_forward(model,d); out=[]; forbidden=[];intent=[]
 for c in d.contact:
  g1=name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1);g2=name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2);b1=name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[c.geom1]);b2=name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[c.geom2]);pair=f'{b1}|{b2}'; rec={'geoms':[g1,g2],'bodies':[b1,b2],'distance_m':float(c.dist)};out.append(rec)
  is_c=('right_hand_middle_1_link' in (b1,b2));is_ring=g1.startswith('ring_segment') or g2.startswith('ring_segment')
  adjacent=(b1.startswith('right_hand_middle') and b2.startswith('right_hand_middle'))
  if allow_hook and is_c and is_ring: intent.append(rec)
  elif not adjacent and (is_ring or 'phone_collision' in (g1,g2) or 'table_collision' in (g1,g2) or ('torso' in pair) or (b1.startswith('left_') and b2.startswith('right_')) or (b2.startswith('left_') and b1.startswith('right_'))): forbidden.append(rec)
 return out,forbidden,intent

def render_geometry():
 for view in ('front','side','top'):
  fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d');th=np.linspace(0,2*np.pi,200);ax.plot(RING_CENTER[0]+INNER*np.cos(th),np.full_like(th,RING_CENTER[1]),RING_CENTER[2]+INNER*np.sin(th),'g');ax.plot(RING_CENTER[0]+OUTER*np.cos(th),np.full_like(th,RING_CENTER[1]),RING_CENTER[2]+OUTER*np.sin(th),'k');ax.scatter(*PHONE_CENTER,c='b');ax.quiver(*RING_CENTER,0,.05,0,color='r');ax.set_title('ACTIVE COLLISION GEOMETRY | DIAGNOSTIC SIMULATION');ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');
  if view=='front':ax.view_init(0,-90)
  elif view=='side':ax.view_init(0,0)
  else:ax.view_init(90,-90)
  fig.savefig(OUT/f'geometry_{view}.png',dpi=160);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--max-nfev',type=int,default=100);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 info,start,model,tmp,rarm,rq,q0,layout=context();audit,gid=audit_geometry(model,q0,rq);render_geometry()
 with np.load(SOURCE) as z: raw=z['optimized_action'];ts=z['timestamp'];fps=float(z['fps'])
 am,_=epi.aloha.load_validated_model(epi.ALOHA_XML);aq,_=epi.aloha.mapped_qpos(raw);fk=epi.aloha.fk(am,aq);rp=fk['right_position_m'];approach=rp[329]-rp[326];removal=rp[341]-rp[329]
 prior={'frames':{'right_accessory_grasp_start':326,'accessory_detachment_start':329,'accessory_removed':341},'approach_vector_aloha_m':approach,'removal_vector_aloha_m':removal,'approach_duration_s':3/fps,'removal_duration_s':12/fps,'gripper_values':raw[[326,329,341],13]};dump(OUT/'aloha_right_motion_prior_metrics.json',prior)
 grid={'radial_angle_deg':list(np.linspace(0,337.5,16)),'preinsert_distance_m':[.03,.055,.08],'insertion_depth_m':[.002,.004,.006],'wrist_rpy_offset_deg':[[0,0,0],[5,0,0],[-5,0,0],[0,5,0],[0,-5,0]],'C_flexion_fraction':[0,.5,1.0],'AB_posture':['open','relaxed'],'removal_distance_m':[.03,.05,.08]};dump(OUT/'insertion_candidate_grid.json',grid)
 if not a.execute: print('DRY_RUN');return 0
 # Search PREINSERT using all 16 radial angles and the full requested 30--80 mm range.
 hand0=q0[rq].copy();results=[];best=None
 for ang in grid['radial_angle_deg']:
  ar=math.radians(ang);rad=np.array([math.cos(ar),0,math.sin(ar)]);Rt=desired_R(rad)
  for dist in grid['preinsert_distance_m']:
   for rpy in grid['wrist_rpy_offset_deg']:
    Rcandidate=Rt@Rotation.from_euler('xyz',rpy,degrees=True).as_matrix()
    target=RING_CENTER+np.array([0,dist,0])+rad*(INNER-max(audit['right_C']['pad_half_extents_m'])-.001)
    q,sol=solve_pose(model,q0,rarm,rq,target,Rcandidate,hand0,a.max_nfev);d=mujoco.MjData(model);d.qpos[:]=q;mujoco.mj_forward(model,d);tip=d.geom_xpos[gid].copy();cs,forbid,_=contacts(model,q)
    rec={'angle_deg':ang,'preinsert_distance_m':dist,'wrist_rpy_offset_deg':rpy,'optimizer_success':bool(sol.success),'optimizer_status':int(sol.status),'optimizer_nfev':int(sol.nfev),'tip_error_m':float(np.linalg.norm(tip-target)),'forbidden_contacts':forbid,'cost':float(sol.cost),'q':q}
    rec['valid']=bool(sol.success and rec['tip_error_m']<=.003 and not forbid);results.append(rec)
    if rec['valid'] and (best is None or rec['tip_error_m']<best['tip_error_m']):best=rec
 dump(OUT/'insertion_candidate_results.json',{'candidate_count':len(results),'preinsert_valid_count':sum(x['valid'] for x in results),'candidates':[{k:serial(v) for k,v in x.items() if k!='q'} for x in results]})
 if best is None:
  failed=min(results,key=lambda x:x['tip_error_m'])
  status='BLOCKED_PREINSERT_IK';selected={'status':status,'selected_candidate':None,'evaluated_candidates':len(results),'valid_candidates':0,'best_failed_candidate':{'radial_angle_deg':failed['angle_deg'],'preinsert_distance_m':failed['preinsert_distance_m'],'wrist_rpy_offset_deg':failed['wrist_rpy_offset_deg'],'tip_error_m':failed['tip_error_m'],'forbidden_contact_count':len(failed['forbidden_contacts']),'optimizer_status':failed['optimizer_status'],'optimizer_nfev':failed['optimizer_nfev']},'full_v3_resume':False}
 else:
  # Do not claim later states without a collision-constrained annulus solve.
  status='BLOCKED_CONTINUOUS_INSERTION_COLLISION';selected={'status':status,'selected_preinsert':{k:serial(v) for k,v in best.items() if k!='q'},'selected_candidate':None,'reason':'PREINSERT reachable; collision-constrained ENTRY/INSERTED/HOOKED/REMOVED solver has not produced a passing candidate.','full_v3_resume':False}
  np.savez_compressed(OUT/'preinsert_state.npz',right_arm_q=best['q'][rarm],right_dex3_q=best['q'][rq],full_qpos=best['q'],phone_pose=np.r_[PHONE_CENTER,1,0,0,0],ring_pose=np.r_[RING_CENTER,1,0,0,0],diagnostic_only=True,real_robot_command_allowed=False)
 dump(OUT/'selected_insertion_candidate.json',selected);dump(OUT/'continuous_collision_results.json',{'status':status,'segments_checked':[],'minimum_required_substeps':50,'max_fingertip_step_m':None,'hard_geometry_gate':'NOT_PASSED'});dump(OUT/'hook_contact_metrics.json',{'status':'NOT_EVALUATED_NO_CONTINUOUS_INSERTION_CANDIDATE','inner_rim_contact_confirmed':False});dump(OUT/'hook_transform_drift.json',{'status':'NOT_EVALUATED','T_C_hook_from_accessory':None})
 (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nMPLCONFIGDIR=/tmp/right_c_ring_mpl PYTHONPATH=tools python tools/run_right_c_ring_insertion_feasibility.py --execute\n')
 report=f'''# Right-C ring insertion feasibility\n\nStatus: `{status}`\n\nActive 12-segment collision ring and right-middle distal collision geometry were used. PREINSERT candidates: {len(results)}, valid: {sum(x['valid'] for x in results)}. No full-task trajectory was generated.\n\nSIMULATION ONLY — NO REAL ROBOT COMMANDS — NO DDS OR PUBLISHER\nRIGHT C RING INSERTION IS A DIAGNOSTIC SIMULATION MODULE\nALOHA RIGHT-HAND MOTION PRESERVED AS A PRIOR\nREAL G1 SAFETY NOT_PERFORMED\n''';(OUT/'report.md').write_text(report);(OUT/'report').mkdir(exist_ok=True);(OUT/'report/index.html').write_text(f'<h1>{status}</h1><p>Diagnostic simulation only. No full trajectory generated.</p>')
 print(status);return 0 if status=='RIGHT_C_INSERTION_MODULE_READY' else 2
if __name__=='__main__':raise SystemExit(main())
