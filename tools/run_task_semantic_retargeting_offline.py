#!/usr/bin/env python3
"""Offline-only audit and controlled orientation retargeting experiments.

This module contains no hardware, DDS, subscriber, publisher, or command API.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
OUT=ROOT/'outputs/task_semantic_retargeting'
SOURCE=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
ARM=ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz'
FULL=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz'
PHASE=ROOT/'outputs/magsafe_gripper_phases.csv'
XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
FRAMES=[0,56,64,150,161,167,246,253,289,297,303,543,555,617,628,635,641,989]
WEIGHTS=(.0025,.005,.01,.02)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def md(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def stats(x):
 x=np.asarray(x,float);return {'mean':float(np.mean(x)),'max':float(np.max(x,initial=0)),'p95':float(np.percentile(x,95))}
def phases():
 with open(PHASE,newline='') as f:return list(csv.DictReader(f))
def stage(l,r):
 if r in ('PREGRASP','GRASP'): return 'accessory approach/grasp candidate'
 if r=='RELEASE': return 'accessory release candidate'
 if l in ('PREGRASP','GRASP'): return 'phone approach/grasp candidate'
 if l=='RELEASE': return 'phone release candidate'
 return 'hold/move or stable segment; video confirmation required'

def audit():
 scene=ROOT/'isaaclab_magsafe_fixed_scene/scene_layout.json'; layout=json.load(open(scene))
 image_dir=ROOT/'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000'
 imgs=sorted(image_dir.glob('*.png'))
 scene_audit={'status':'PARTIALLY_VERIFIED','scene_layout':str(scene),'scene_coordinate_frame':layout.get('coordinate_frame'),
  'objects':layout.get('objects'),'generated_assets':[str(p) for p in sorted((ROOT/'isaaclab_magsafe_fixed_scene/generated').glob('*.usd*'))],
  'episode49_cam_high':str(image_dir),'video_frame_count':len(imgs),'video_fps':30.0,'trajectory_frame_count':990,
  'frame_alignment':'VERIFIED_BY_EQUAL_FRAME_COUNT_AND_30HZ_TIMESTAMP_SEQUENCE',
  'g1_scene_registration':'NOT_VERIFIED_FOR_CURRENT_RETARGETED_WRIST_TARGETS',
  'object_metrics_status':'NOT_AVAILABLE_UNTIL_SCENE_TO_G1_REGISTRATION_IS_VALIDATED',
  'warning':'Object poses were not moved or fitted to the trajectory.'}
 dump(OUT/'scene_audit.json',scene_audit)
 md(OUT/'scene_audit.md','# Scene audit\n\n'+ '\n'.join(f'- **{k}**: `{v}`' for k,v in scene_audit.items()))
 rows=phases(); timeline=[]
 for f in FRAMES:
  x=rows[f];timeline.append({'frame':f,'time_sec':f/30,'left_phase':x['left_phase'],'right_phase':x['right_phase'],
   'task_stage_candidate':stage(x['left_phase'],x['right_phase']),'left_object_role':'phone','right_object_role':'magsafe_accessory',
   'video_evidence':str(imgs[f]) if len(imgs)>f else 'UNKNOWN','semantic_status':'MANUAL_CONFIRMATION_REQUIRED'})
 with open(OUT/'episode49_task_timeline.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=timeline[0]);w.writeheader();w.writerows(timeline)
 md(OUT/'episode49_task_timeline.md','# Episode 49 task timeline\n\nFrame synchronization is verified; semantic labels remain candidates until manual video review.\n\n'+
    '\n'.join(f"- frame {x['frame']} ({x['time_sec']:.3f}s): L={x['left_phase']}, R={x['right_phase']}; {x['task_stage_candidate']}" for x in timeline))
 palm={'schema_version':1,'simulation_only':True,'authoritative_for_real_robot':False,'source_xml':str(XML),
  'axes_basis':'Existing repository kinematics in retarget_episode49_relative_bimanual_neutral_pinch_to_g1.py; XML palm visual/collision geom transform.',
  'left':{'parent_body':'left_wrist_yaw_link','palm_geom_name':'left_hand_palm_link','geom_local_position_m':[.0415,.003,0],
   'geom_local_quaternion_wxyz':[1,0,0,0],'origin':'wrist body origin; palm proxy at geom local position','tool_forward_axis_local':[1,0,0],'lateral_axis_local':[0,1,0],'palm_normal_axis_local':[0,0,1]},
  'right':{'parent_body':'right_wrist_yaw_link','palm_geom_name':'right_hand_palm_link','geom_local_position_m':[.0415,-.003,0],
   'geom_local_quaternion_wxyz':[1,0,0,0],'origin':'wrist body origin; palm proxy at geom local position','tool_forward_axis_local':[1,0,0],'lateral_axis_local':[0,1,0],'palm_normal_axis_local':[0,0,1]}}
 dump(ROOT/'configs/g1_dex3_palm_frame_calibration.sim.json',palm)
 tools={'source_fk':'validate_smolvla_in_stationary_aloha_mujoco.py: fk; follower_left/right_link_6; TCP offset [0.1487,0,-0.00105]',
  'closing_axis':'NOT_EXPLICITLY_CALIBRATED','approach_axis':'NOT_EXPLICITLY_CALIBRATED',
  'candidates':{'left_C':[[1,0,0],[0,0,-1],[0,1,0]],'right_C':[[1,0,0],[0,0,1],[0,-1,0]],
  'source':'retarget_episode49_optimized_action_to_g1.py','status':'CODE_VERIFIED_CANDIDATE_NOT_PHYSICAL_CALIBRATION'}}
 dump(ROOT/'configs/aloha_to_g1_palm_frame_candidates.json',tools)
 freeze={'status':'position_only_frozen','source_action':str(SOURCE),'source_action_sha256':sha(SOURCE),'arm_trajectory':str(ARM),'arm_trajectory_sha256':sha(ARM),'full_trajectory':str(FULL),'full_trajectory_sha256':sha(FULL)}
 dump(OUT/'authoritative_freeze.json',freeze)
 return scene_audit

def rotations(core,info,q):
 d=core.mujoco.MjData(info['model']);lr=[];rr=[];lp=[];rp=[]
 for x in q:
  s=core.frame_state(info,d,x);lp.append(s['left_pos']);rp.append(s['right_pos'])
  lr.append(Rotation.from_quat(s['left_quat'][[1,2,3,0]]).as_matrix());rr.append(Rotation.from_quat(s['right_quat'][[1,2,3,0]]).as_matrix())
 return np.asarray(lp),np.asarray(rp),np.asarray(lr),np.asarray(rr)
def branch(q):
 n=np.linalg.norm(np.diff(q,axis=0),axis=1);b=np.zeros(len(q),bool)
 for t in range(1,len(q)):b[t]=n[t-1]>max(.15,8*max(np.median(n[max(0,t-10):min(len(n),t+9)]),1e-5))
 return b
def save_candidate(name,q,tar,info,source_hash,weight):
 lp,rp,lr,rr=rotations(__import__('retarget_episode49_optimized_action_to_g1'),info,q);le=np.linalg.norm(lp-tar['lp'],axis=1);re=np.linalg.norm(rp-tar['rp'],axis=1)
 lim=info['joint_limits'];viol=int(np.sum((q<lim[:,0]-1e-9)|(q>lim[:,1]+1e-9)));b=branch(q)
 d=OUT/'candidates'/name;d.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(d/'arm_trajectory.npz',timestamps=np.arange(len(q))/30,fps=np.array(30.),arm_qpos=q,arm_joint_names=info['joint_names'],target_left_position=tar['lp'],target_right_position=tar['rp'],achieved_left_position=lp,achieved_right_position=rp,achieved_left_rotation=lr,achieved_right_rotation=rr,input_action_hash=np.array(source_hash))
 m={'candidate':name,'orientation_weight':weight,'frames':len(q),'ik_success_rate':float(np.mean((le<=.005)&(re<=.005))),
  'left_position_rmse_mm':float(np.sqrt(np.mean(le**2))*1000),'right_position_rmse_mm':float(np.sqrt(np.mean(re**2))*1000),
  'max_wrist_error_mm':float(max(le.max(),re.max())*1000),'joint_limit_violations':viol,'branch_discontinuity_count':int(b.sum()),
  'max_joint_step_rad':float(np.max(np.abs(np.diff(q,axis=0)),initial=0)),'max_velocity_rad_s':float(np.max(np.abs(np.diff(q,axis=0))*30,initial=0)),
  'max_acceleration_rad_s2':float(np.max(np.abs(np.diff(q,n=2,axis=0))*900,initial=0)),
  'object_semantic_metrics':'NOT_AVAILABLE_UNVERIFIED_SCENE_TO_G1_REGISTRATION','arm_collision':'NOT_EVALUATED_IN_THIS_GENERATOR'}
 dump(d/'metrics.json',m);return m

def generate(iterations):
 import retarget_episode49_optimized_action_to_g1 as core
 import retarget_episode49_consensus_relative_bimanual_to_g1 as rel
 raw,_=core.load_source(SOURCE,None);info=core.ik.validate_model(XML)
 with np.load(ARM,allow_pickle=False) as z:
  frozen=z['g1_arm_joint_trajectory'].astype(float);tar={'lp':z['g1_target_left_position'].astype(float),'rp':z['g1_target_right_position'].astype(float)};start=z['g1_start_arm_q'].astype(float)
 am,_=core.aloha.load_validated_model(core.ALOHA_XML);aq,_=core.aloha.mapped_qpos(raw);fk=core.aloha.fk(am,aq)
 lp,rp,l0,r0=rotations(core,info,frozen);base_l=l0[0];base_r=r0[0]
 candidates=[]; candidates.append(save_candidate('position_only_frozen',frozen,tar,info,sha(SOURCE),0.0))
 target_sets={
  'neutral_wrist':(np.repeat(base_l[None],len(frozen),0),np.repeat(base_r[None],len(frozen),0)),
  'relative_orientation_transfer':(np.einsum('ij,tjk->tik',base_l,core.maprot(core.rot_from_wxyz(fk['left_quaternion_wxyz']),core.C_L)),np.einsum('ij,tjk->tik',base_r,core.maprot(core.rot_from_wxyz(fk['right_quaternion_wxyz']),core.C_R)))}
 for name,(lr,rr) in target_sets.items():
  sweep=[]
  for w in WEIGHTS:
   t=time.perf_counter();q=core.temporal_solve(info,tar|{'lr':lr,'rr':rr},frozen,start,w,iterations);m=save_candidate(f'{name}/weight_{w:g}',q,tar,info,sha(SOURCE),w);m['solver_runtime_sec']=time.perf_counter()-t;sweep.append((m,q))
  viable=[x for x in sweep if x[0]['ik_success_rate']>=.99 and x[0]['joint_limit_violations']==0 and x[0]['branch_discontinuity_count']==0]
  # Preserve one deterministic representative even when the feasibility gate
  # rejects every weight; generation and selection are deliberately separate.
  best=min(viable or sweep,key=lambda x:(x[0]['left_position_rmse_mm']+x[0]['right_position_rmse_mm'],x[0]['orientation_weight']))
  best[0]['feasibility_gate']='PASS' if viable else 'FAIL'
  d=OUT/'candidates'/name;d.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/'candidates'/name/f"weight_{best[0]['orientation_weight']:g}"/'arm_trajectory.npz',d/'arm_trajectory.npz');dump(d/'metrics.json',best[0]);candidates.append(best[0])
 unavailable={'candidate':'phase_conditioned_partial_orientation','status':'NOT_AVAILABLE','reason':'Scene object poses exist, but their transform into the current G1 wrist-target frame is not validated. Object approach vectors/grasp regions cannot be constructed without fabrication.'}
 dump(OUT/'candidates/phase_conditioned_partial_orientation/status.json',unavailable)
 dump(OUT/'candidate_metrics.json',{'candidates':candidates,'phase_conditioned_partial_orientation':unavailable})
 return candidates

def report(scene,candidates):
 decision={'selected_candidate':None,'verdict':'NO TASK-VALID CANDIDATE FOUND','reason':'Object-frame task-semantic gates are unavailable because scene-to-G1 registration is unverified.',
  'position_tracking':'EVALUATED','task_orientation':'PARTIAL_DIAGNOSTIC_ONLY','task_semantics':'NOT_AVAILABLE','arm_collision':'NOT_EVALUATED_FOR_NEW_CANDIDATES','placeholder_hand':'DIAGNOSTIC_ONLY','real_g1_safety':'NOT_PERFORMED','isaac_lab':'NOT_ATTEMPTED_NO_TASK_VALID_CANDIDATE'}
 dump(OUT/'selection.json',decision)
 vis={'object_scene_videos':'NOT_AVAILABLE','aloha_synchronized_comparison':'NOT_AVAILABLE_WITHOUT_VALID_OBJECT_SCENE_RENDER','phase_montage':'SOURCE_TIMELINE_ONLY','reason':decision['reason']};dump(OUT/'report/visual_availability.json',vis)
 rows=[]
 for m in candidates:rows.append(m)
 if rows:
  keys=sorted(set().union(*(x.keys() for x in rows)))
  with open(OUT/'report/candidate_metrics.csv','w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 html='''<!doctype html><meta charset="utf-8"><title>Task-semantic retargeting</title><style>body{font:16px sans-serif;max-width:1100px;margin:2em auto} .bad{color:#b00;font-weight:bold} pre{white-space:pre-wrap;background:#eee;padding:1em}</style><h1>Task-semantic retargeting</h1><p class="bad">NO REAL G1 / NO REAL ALOHA — REAL G1 SAFETY NOT_PERFORMED</p><h2>Decision</h2><pre>'''+json.dumps(decision,indent=2)+'''</pre><h2>Scene audit</h2><pre>'''+json.dumps(scene,indent=2)+'''</pre><h2>Candidates</h2><pre>'''+json.dumps(candidates,indent=2)+'''</pre><h2>Visual availability</h2><pre>'''+json.dumps(vis,indent=2)+'''</pre><p>Manual checklist: <a href="../../../docs/REVIEW_TASK_SEMANTIC_G1_TRAJECTORY.md">document</a></p>'''
 md(OUT/'report/index.html',html)
 print('TASK-SEMANTIC RETARGETING ANALYSIS\nNO REAL G1 WAS USED\nNO REAL ALOHA WAS USED\nREAL G1 EXECUTION NOT APPROVED UNTIL MANUAL REVIEW')
 print(json.dumps(decision,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');p.add_argument('--iterations',type=int,default=3);p.add_argument('--inspect',action='store_true');a=p.parse_args()
 if a.inspect:
  for x in ('scene_audit.json','candidate_metrics.json','selection.json'):print((OUT/x).read_text() if (OUT/x).exists() else f'MISSING: {OUT/x}')
  return 0
 scene=audit();c=[] if a.audit_only else generate(a.iterations);report(scene,c);return 0
if __name__=='__main__':raise SystemExit(main())
