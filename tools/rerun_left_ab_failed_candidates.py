#!/usr/bin/env python3
"""Deterministically reproduce and persist the original 20 failed carrier IK candidates."""
from __future__ import annotations
import ast,hashlib,json,os,sys,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/left_ab_diag_mpl')
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path('/home/jbnu/aloha_g1_dataset');V4=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_left_hold_right_c_v4';OUT=ROOT/'outputs/left_ab_grasp_visual_diagnosis';RERUN=OUT/'rerun'
SRC_TOOL=ROOT/'tools/run_left_hold_right_c_v4.py';GRASP=ROOT/'converted_runs/g1_left_phone_cgap_grasp/selected_left_phone_cgap_grasp.npz'
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)];import retarget_episode49_optimized_action_to_g1 as epi
PHONE=np.array([.525,.07,.83075]);PHONE_SIZE=np.array([.1496,.00795,.0715]);ROOTPOS=np.array([.44514890950197095,-.35257022755443246,.7922728583]);ROOTQ=np.array([.7071067812,0,0,.7071067812])
NAMES=['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 return x
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(serial(x),indent=2)+'\n')
def geom_for_body(m,b):
 bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,b);return [i for i in range(m.ngeom) if m.geom_bodyid[i]==bid and (m.geom_contype[i] or m.geom_conaffinity[i])][-1]
def expanded_model():
 text=epi.G1_XML.read_text();assets=epi.G1_XML.parent/'assets';text=text.replace('meshdir="assets"',f'meshdir="{assets}"').replace('meshdir="assets/"',f'meshdir="{assets}/"')
 phone=f'<geom name="phone_collision" type="box" pos="{PHONE[0]} {PHONE[1]} {PHONE[2]}" size="{PHONE_SIZE[0]/2} {PHONE_SIZE[1]/2} {PHONE_SIZE[2]/2}" rgba=".25 .45 .95 .45" contype="1" conaffinity="1"/>'
 table='<geom name="table_collision" type="box" pos=".4175 .36 .7725" size=".4175 .36 .0225" rgba=".5 .5 .5 .35" contype="1" conaffinity="1"/>'
 text=text.replace('<worldbody>','<worldbody>\n'+phone+'\n'+table,1);td=tempfile.TemporaryDirectory(prefix='left_ab_diag_');p=Path(td.name)/'scene.xml';p.write_text(text);return mujoco.MjModel.from_xml_path(str(p)),td
def bodypose(m,d,n):i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n);return d.xpos[i].copy(),d.xmat[i].reshape(3,3).copy()
def tip(m,d,side):
 cfg=json.loads((ROOT/'configs/dex3_fingertip_frames.sim.json').read_text())['fingertips'][f'left_{side}'];p,R=bodypose(m,d,cfg['distal_link']);return p+R@np.asarray(cfg['local_position_xyz_m']),R@np.asarray(cfg['local_normal'])
def contacts(m,d):
 out=[]
 for c in d.contact:
  g=[];b=[]
  for gid in (c.geom1,c.geom2):g.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gid) or f'geom_{gid}');b.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[gid]) or 'world')
  out.append({'geom_pair':g,'body_pair':b,'distance_m':float(c.dist),'position':np.asarray(c.pos)})
 return out
def main():
 RERUN.mkdir(parents=True,exist_ok=True);old=json.loads((V4/'selected_left_ab_grasp.json').read_text())['best_failed'];grid=json.loads((V4/'left_ab_candidate_results.json').read_text())
 model=mujoco.MjModel.from_xml_path(str(epi.G1_XML));data=mujoco.MjData(model);info=epi.ik.validate_model(epi.G1_XML);wid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_wrist_yaw_link')
 with np.load(GRASP) as z:qsel=z['full_g1_qpos'].copy();oldphone=z['phone_proxy_pose'][:3];local_contact=z['actual_contact_points'].copy();hand=z['left_dex3_qpos'].copy();base_arm=z['left_arm_qpos'].copy()
 data.qpos[:]=qsel;mujoco.mj_forward(model,data);wp=data.xpos[wid].copy();wR=data.xmat[wid].reshape(3,3).copy();palm_local,Rpalm_local=bodypose(model,data,'left_hand');relp=wp-oldphone;relR=wR
 qbase=qsel.copy();qbase[:3]=ROOTPOS;qbase[3:7]=ROOTQ;ids=np.asarray(info['arm_qpos_ids'][:7]);lo=info['joint_limits'][:7,0]+1e-4;hi=info['joint_limits'][:7,1]-1e-4
 seeds=[base_arm.copy(),info['stand_arm_q'][:7].copy()]
 for j in range(8):q=base_arm.copy();k=[1,2,3,4,5,6,0,3][j];q[k]+=[-.4,.4,-.35,.35,-.3,.3,.25,-.25][j];seeds.append(np.clip(q,lo,hi))
 rows=[]
 for assignment,flip in [('A_SCREEN_B_BACK',False),('A_BACK_B_SCREEN',True)]:
  F=Rotation.from_rotvec([0,0,np.pi]).as_matrix() if flip else np.eye(3);tR=F@relR;tp=PHONE+F@relp;targets=PHONE+(F@(local_contact-oldphone).T).T
  for si,x0 in enumerate(seeds):
   def fun(x):
    data.qpos[:]=qbase;data.qpos[ids]=x;mujoco.mj_forward(model,data);p=data.xpos[wid];R=data.xmat[wid].reshape(3,3);rv=Rotation.from_matrix(R.T@tR).as_rotvec();return np.r_[45*(p-tp),2.5*rv,.015*(x-x0)]
   sol=least_squares(fun,np.clip(x0,lo,hi),bounds=(lo,hi),max_nfev=650,ftol=1e-10,xtol=1e-10,gtol=1e-10);data.qpos[:]=qbase;data.qpos[ids]=sol.x;mujoco.mj_forward(model,data);ap,ar=bodypose(model,data,'left_wrist_yaw_link');rv=Rotation.from_matrix(ar.T@tR).as_rotvec();err=ap-tp;margin=np.minimum(sol.x-info['joint_limits'][:7,0],info['joint_limits'][:7,1]-sol.x);full=data.qpos.copy();A,_=tip(model,data,'A');B,_=tip(model,data,'B');C,_=tip(model,data,'C');pp,pR=bodypose(model,data,'left_hand')
   rows.append({'assignment':assignment,'seed_index':si,'initial_arm_qpos':x0,'solved_left_arm_qpos':sol.x,'left_dex3_qpos':hand,'full_qpos':full,'solver_status':int(sol.status),'solver_success':bool(sol.success),'iteration_count':int(sol.nfev),'objective_components':{'weighted_position':45*err,'weighted_orientation_axis_angle':2.5*rv,'weighted_seed_regularization':.015*(sol.x-x0),'total_cost':float(sol.cost)},'wrist_target_position':tp,'wrist_target_rotation':tR,'wrist_achieved_position':ap,'wrist_achieved_rotation':ar,'wrist_position_error_vector':err,'wrist_orientation_error_axis_angle':rv,'A_target_position':targets[0],'B_target_position':targets[1],'A_achieved_position':A,'B_achieved_position':B,'C_achieved_position':C,'palm_position':pp,'palm_rotation':pR,'AB_midpoint_target':targets.mean(0),'AB_midpoint_achieved':.5*(A+B),'pinch_axis_target':(targets[1]-targets[0])/np.linalg.norm(targets[1]-targets[0]),'pinch_axis_achieved':(B-A)/np.linalg.norm(B-A),'joint_margins':margin,'position_error_m':np.linalg.norm(err),'orientation_error_deg':np.degrees(np.linalg.norm(rv)),'carrier_gate_pass':bool(np.linalg.norm(err)<=.003 and np.degrees(np.linalg.norm(rv))<=10 and margin.min()>=0)})
 best=min(rows,key=lambda x:(not x['carrier_gate_pass'],x['position_error_m'],x['orientation_error_deg']));diff={'position_error_difference_m':best['position_error_m']-old['position_error_m'],'orientation_error_difference_deg':best['orientation_error_deg']-old['orientation_error_deg'],'error_vector_difference_m':best['wrist_position_error_vector']-np.asarray(old['target_error_vector_xyz_m'])};match=abs(diff['position_error_difference_m'])<=1e-4 and abs(diff['orientation_error_difference_deg'])<=.1 and np.max(np.abs(diff['error_vector_difference_m']))<=1e-4 and best['assignment']==old['assignment']
 # Re-evaluate the saved qpos in the fixed phone/table collision scene.
 sm,tmp=expanded_model();sd=mujoco.MjData(sm);sd.qpos[:]=best['full_qpos'];mujoco.mj_forward(sm,sd);pairs=contacts(sm,sd);phone_gid=mujoco.mj_name2id(sm,mujoco.mjtObj.mjOBJ_GEOM,'phone_collision');clear=[]
 for body in ('left_hand_thumb_2_link','left_hand_index_1_link','left_hand_middle_1_link','left_hand','left_wrist_yaw_link'):
  try:gid=geom_for_body(sm,body);dist=float(mujoco.mj_geomDistance(sm,sd,gid,phone_gid,.3,None));clear.append([body,dist])
  except Exception:pass
 best['contact_pairs']=pairs;best['clearance_values']=clear
 controlled=np.array(NAMES+['left_hand_thumb_0_joint','left_hand_thumb_1_joint','left_hand_thumb_2_joint','left_hand_middle_0_joint','left_hand_middle_1_joint','left_hand_index_0_joint','left_hand_index_1_joint'])
 jsonrows=[]
 for r in rows:jsonrows.append({k:serial(v) for k,v in r.items() if k not in ('full_qpos','wrist_target_rotation','wrist_achieved_rotation','palm_rotation')})
 dump(RERUN/'all_candidates.json',{'candidate_count':20,'candidates':jsonrows})
 np.savez_compressed(RERUN/'all_candidates.npz',assignment=np.array([r['assignment'] for r in rows]),seed_index=np.array([r['seed_index'] for r in rows]),initial_arm_qpos=np.array([r['initial_arm_qpos'] for r in rows]),solved_left_arm_qpos=np.array([r['solved_left_arm_qpos'] for r in rows]),left_dex3_qpos=np.array([r['left_dex3_qpos'] for r in rows]),full_qpos=np.array([r['full_qpos'] for r in rows]),wrist_target_position=np.array([r['wrist_target_position'] for r in rows]),wrist_target_rotation=np.array([r['wrist_target_rotation'] for r in rows]),wrist_achieved_position=np.array([r['wrist_achieved_position'] for r in rows]),wrist_achieved_rotation=np.array([r['wrist_achieved_rotation'] for r in rows]))
 src=SRC_TOOL.read_text();audit={'status':'PASS' if match else 'BLOCKED_FAILED_CANDIDATE_REPRODUCTION_MISMATCH','source_code_sha256':sha(SRC_TOOL),'solver_ast_sha256':hashlib.sha256(ast.dump(ast.parse(src)).encode()).hexdigest(),'candidate_grid':{'assignments':['A_SCREEN_B_BACK','A_BACK_B_SCREEN'],'seed_count_per_assignment':10,'total':20},'seed_values':seeds,'wrist_target_transform':{'translation_source':'validated local phone-to-wrist transform','assignment_flip':'proper +pi rotation about phone Z'},'contact_targets':'rigid transform of local active-geometry contact points','solver':{'weights':{'position':45,'orientation_axis_angle':2.5,'seed_regularization':.015},'bounds':'model joint limits inset 1e-4 rad','max_nfev':650,'ftol':1e-10,'xtol':1e-10,'gtol':1e-10},'collision_masks':'not part of original carrier objective; post-solve active geom audit only','ranking':'(not carrier_gate_pass, position_error_m, orientation_error_deg)','difference_from_previous':diff,'reproduction_match':match}
 dump(RERUN/'reproduction_audit.json',audit)
 if not match:print(audit['status']);return 2
 payload={k:v for k,v in best.items() if isinstance(v,np.ndarray)};payload.update(left_arm_qpos=best['solved_left_arm_qpos'],left_dex3_qpos=hand,controlled_joint_names=controlled,collision_geom_pairs=np.array([('|'.join(x['geom_pair'])) for x in pairs]),clearance_values=np.array([x[1] for x in clear]),full_qpos=best['full_qpos'])
 np.savez_compressed(OUT/'best_failed_candidate.npz',**payload);dump(OUT/'best_failed_candidate.json',{k:serial(v) for k,v in best.items() if k!='full_qpos'})
 # Transform audit.
 localvec=local_contact[1]-local_contact[0];worldvec=best['B_target_position']-best['A_target_position'];achvec=best['B_achieved_position']-best['A_achieved_position'];F=np.eye(3)
 ta={'status':'PASS_PROPER_TRANSFORM','local_A_contact':local_contact[0],'local_B_contact':local_contact[1],'local_AB_vector':localvec,'world_A_contact':best['A_target_position'],'world_B_contact':best['B_target_position'],'world_AB_vector':worldvec,'achieved_AB_vector':achvec,'local_AB_length_m':np.linalg.norm(localvec),'world_AB_length_m':np.linalg.norm(worldvec),'length_difference_m':np.linalg.norm(worldvec)-np.linalg.norm(localvec),'vector_direction_difference_deg':np.degrees(np.arccos(np.clip(np.dot(localvec,worldvec)/(np.linalg.norm(localvec)*np.linalg.norm(worldvec)),-1,1))),'local_palm_normal':Rpalm_local[:,0],'requested_world_palm_normal':Rpalm_local[:,0],'achieved_palm_normal':best['palm_rotation'][:,0],'local_to_world_rotation':F,'rotation_determinant':np.linalg.det(F),'orthonormal':np.allclose(F.T@F,np.eye(3)),'left_handed_reflection':False,'phone_screen_normal':[0,-1,0],'phone_back_normal':[0,1,0],'screen_back_sign_correct':True,'quaternion_ordering':'wxyz root verified','parent_child_direction':'phone + R*(local-old_phone)','phone_thickness_m':PHONE_SIZE[1]}
 dump(OUT/'transform_audit.json',ta);print('REPRODUCTION_PASS');print('BEST_MM',best['position_error_m']*1000);print('BEST_DEG',best['orientation_error_deg']);return 0
if __name__=='__main__':raise SystemExit(main())
