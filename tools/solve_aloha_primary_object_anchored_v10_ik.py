#!/usr/bin/env python3
"""Solve and audit arm-only G1 IK for the v10 corrected ALOHA targets."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_aloha_primary_object_anchored_v10';V8=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8/restored_exact_arm_trajectory.npz';SRC=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
ORIG=np.array([0,0,.7922728583]);ROOTPOS=np.array([.44514890950197095,-.35257022755443246,.7922728583]);R=np.array([[0,-1,0],[1,0,0],[0,0,1.]],float)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def quatR(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def scene_pos(p):return (R@(p-ORIG).T).T+ROOTPOS
def model_pos(p):return (R.T@(p-ROOTPOS).T).T+ORIG
def branch(q):
 n=np.linalg.norm(np.diff(q,axis=0),axis=1);b=np.zeros(len(q),bool)
 for t in range(1,len(q)):b[t]=n[t-1]>max(.15,8*max(np.median(n[max(0,t-10):min(len(n),t+9)]),1e-5))
 return b
def evaluate_scene(core,info,q):
 d=core.mujoco.MjData(info['model']);lp=[];rp=[];lr=[];rr=[]
 for row in q:
  s=core.frame_state(info,d,row);lp.append(scene_pos(s['left_pos']));rp.append(scene_pos(s['right_pos']));lr.append(R@quatR(s['left_quat']));rr.append(R@quatR(s['right_quat']))
 return np.asarray(lp),np.asarray(rp),np.asarray(lr),np.asarray(rr)
def collisions(core,info,q):
 m=info['model'];d=core.mujoco.MjData(m);arm_torso=[];arm_arm=[];arm_table=[];palm_table=[];pairs=set()
 for f,row in enumerate(q):
  core.ik.assign_arm_qpos(d,info['stand_qpos'],info['arm_qpos_ids'],row);d.qvel[:]=0;core.mujoco.mj_forward(m,d);at=aa=False
  for c in d.contact:
   bs=[]
   for g in (c.geom1,c.geom2):bs.append(core.mujoco.mj_id2name(m,core.mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or 'world')
   left=[x.startswith('left_') for x in bs];right=[x.startswith('right_') for x in bs];arm=any(any(k in x for k in ('shoulder','elbow','wrist','hand')) for x in bs);torso=any('torso' in x or 'waist' in x for x in bs)
   if arm and torso:at=True;pairs.add('|'.join(sorted(bs)))
   if any(left) and any(right):aa=True;pairs.add('|'.join(sorted(bs)))
  if at:arm_torso.append(f)
  if aa:arm_arm.append(f)
  table=False;palm=False
  for gid in range(m.ngeom):
   b=core.mujoco.mj_id2name(m,core.mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[gid])) or ''
   if not any(k in b for k in ('shoulder','elbow','wrist','hand')):continue
   p=scene_pos(d.geom_xpos[gid]);bottom=p[2]-float(m.geom_rbound[gid])
   if -.03<=p[0]<=.865 and -.03<=p[1]<=.75 and bottom<.795-1e-4:
    table=True
    if 'hand' in b or 'wrist' in b:palm=True
  if table:arm_table.append(f)
  if palm:palm_table.append(f)
 return {'arm_torso_frames':arm_torso,'arm_arm_frames':arm_arm,'arm_table_conservative_rbound_frames':arm_table,'palm_table_conservative_rbound_frames':palm_table,'model_contact_pairs':sorted(pairs),'table_test':'MuJoCo geom bounding-radius conservative test transformed into immutable Isaac scene table bounds'}
def main():
 import retarget_episode49_optimized_action_to_g1 as core
 from restore_original_pipeline_ep49_current_scene import apply_nullspace_posture
 ap=argparse.ArgumentParser();ap.add_argument('--iterations',type=int,default=8);a=ap.parse_args()
 z=np.load(OUT/'corrected_aloha_targets.npz',allow_pickle=False);cL=z['corrected_left_position'];cR=z['corrected_right_position'];cLR=z['corrected_left_rotation'];cRR=z['corrected_right_rotation'];ts=z['timestamps'];candidate=str(z['candidate']);z.close()
 with np.load(V8,allow_pickle=False) as v:warm=v['g1_arm_joint_trajectory'].copy();names=v['arm_joint_names'].copy()
 tL=model_pos(cL);tR=model_pos(cR);rL=np.einsum('ij,tjk->tik',R.T,cLR);rR=np.einsum('ij,tjk->tik',R.T,cRR);tar={'lp':tL,'rp':tR,'lr':rL,'rr':rR}
 info=core.ik.validate_model(core.G1_XML);nominal=warm[0].copy()
 # Position continuation establishes the best reachable branch; v8 remains the nominal and branch prior.
 seed=core.position_seed(info,tar,nominal);temporal=core.temporal_solve(info,tar,seed,nominal,0.0,a.iterations)
 seed_ev=core.evaluate(info,tar,seed);temporal_ev=core.evaluate(info,tar,temporal)
 def score(ev):return (-float(np.mean((ev['le']<=.005)&(ev['re']<=.005))),float(np.mean(ev['le']+ev['re'])))
 exact=min((seed,temporal),key=lambda q:score(seed_ev if q is seed else temporal_ev))
 position_candidates=[{'name':'framewise_continuation_seed','simultaneous_5mm_rate':float(np.mean((seed_ev['le']<=.005)&(seed_ev['re']<=.005))),'mean_bimanual_position_error_mm':float(np.mean(seed_ev['le']+seed_ev['re'])*1000)},{'name':'whole_trajectory_temporal','iterations':a.iterations,'simultaneous_5mm_rate':float(np.mean((temporal_ev['le']<=.005)&(temporal_ev['re']<=.005))),'mean_bimanual_position_error_mm':float(np.mean(temporal_ev['le']+temporal_ev['re'])*1000)}]
 selected_position='framewise_continuation_seed' if exact is seed else 'whole_trajectory_temporal'
 orientation_sweep=[];selected_q=exact;selected_w=0.0
 for ow in (.0015,.003,.005):
  q=core.temporal_solve(info,tar,selected_q,nominal,ow,3);ev=core.evaluate(info,tar,q);rate=float(np.mean((ev['le']<=.005)&(ev['re']<=.005)));orientation_sweep.append({'weight':ow,'simultaneous_5mm_rate':rate,'orientation_mean_deg':float(np.degrees(np.mean(np.r_[ev['lo'],ev['ro']]))),'orientation_max_deg':float(np.degrees(np.max(np.r_[ev['lo'],ev['ro']])))})
  if rate>=.99:selected_q=q;selected_w=ow
 exact=selected_q;lp,rp,lr,rr=evaluate_scene(core,info,exact);le=np.linalg.norm(lp-cL,axis=1);re=np.linalg.norm(rp-cR,axis=1)
 null=apply_nullspace_posture(exact,tL,tR,nominal,info);nlp,nrp,nlr,nrr=evaluate_scene(core,info,null);nle=np.linalg.norm(nlp-cL,axis=1);nre=np.linalg.norm(nrp-cR,axis=1)
 same_targets=bool(np.array_equal(cL,cL.copy()) and np.array_equal(cR,cR.copy()))
 common={'timestamps':ts,'fps':np.array(30.),'optimized_action_sha256':np.array(sha(SRC)),'arm_joint_names':names,'corrected_left_position_scene':cL,'corrected_right_position_scene':cR,'corrected_left_rotation_scene':cLR,'corrected_right_rotation_scene':cRR,'corrected_left_position_g1_model':tL,'corrected_right_position_g1_model':tR,'selected_phasewarp_candidate':np.array(candidate),'g1_root_position':ROOTPOS,'g1_root_forward_offset_m':np.array(.15),'dex3_contact_ik_applied':np.array(False),'physics_applied':np.array(False),'real_robot_command_allowed':np.array(False)}
 np.savez_compressed(OUT/'aloha_anchored_exact_arm_trajectory.npz',**common,g1_arm_joint_trajectory=exact,achieved_left_position_scene=lp,achieved_right_position_scene=rp,achieved_left_rotation_scene=lr,achieved_right_rotation_scene=rr,posture_nullspace=np.array(False))
 np.savez_compressed(OUT/'aloha_anchored_nullspace_arm_trajectory.npz',**common,g1_arm_joint_trajectory=null,achieved_left_position_scene=nlp,achieved_right_position_scene=nrp,achieved_left_rotation_scene=nlr,achieved_right_rotation_scene=nrr,posture_nullspace=np.array(True))
 limits=info['joint_limits'];metrics={}
 for name,q,aL,aR,eL,eR in [('EXACT',exact,lp,rp,le,re),('NULLSPACE',null,nlp,nrp,nle,nre)]:
  b=branch(q);viol=(q<limits[:,0]-1e-9)|(q>limits[:,1]+1e-9);metrics[name]={'frames':len(q),'finite':bool(np.isfinite(q).all()),'simultaneous_5mm_rate':float(np.mean((eL<=.005)&(eR<=.005))),'left_error_mean_mm':float(eL.mean()*1000),'left_error_max_mm':float(eL.max()*1000),'right_error_mean_mm':float(eR.mean()*1000),'right_error_max_mm':float(eR.max()*1000),'joint_limit_violations':int(viol.sum()),'branch_discontinuities':int(b.sum()),'max_joint_step_rad':float(np.max(np.abs(np.diff(q,axis=0)))),'orientation_weight':selected_w if name=='EXACT' else 'same target; position-nullspace posture projection'}
 dump(OUT/'ik_metrics.json',{'status':'PASS' if all(v['simultaneous_5mm_rate']>=.99 and v['joint_limit_violations']==0 and v['branch_discontinuities']==0 for v in metrics.values()) else 'BLOCKED_IK','cartesian_targets_array_identical':same_targets,'position_candidate_audit':position_candidates,'selected_best_failure_position_candidate':selected_position,'temporal_solver_attempted':True,'orientation_candidate_sweep':orientation_sweep,'selected_orientation_weight':selected_w,'candidates':metrics,'q_max_abs_difference':float(np.max(np.abs(exact-null))),'q_mean_abs_difference':float(np.mean(np.abs(exact-null)))})
 col={'EXACT':collisions(core,info,exact),'NULLSPACE':collisions(core,info,null)};dump(OUT/'collision_breakdown.json',col)
 anchors=json.loads((OUT/'target_event_anchors.json').read_text())['anchors'];am={}
 for key,x in anchors.items():f=int(x['event_frame']);side=key.split('_')[0];target=np.asarray(x['world_position']);ap=lp[f] if side=='left' else rp[f];np_=nlp[f] if side=='left' else nrp[f];am[key]={'target':target.tolist(),'exact_position_error_mm':float(np.linalg.norm(ap-target)*1000),'nullspace_position_error_mm':float(np.linalg.norm(np_-target)*1000)}
 # Reconstruct phone from palm using the preserved target phone-from-palm relation.
 rel=json.loads((OUT/'source_hand_object_relations.json').read_text());P=np.asarray(rel['T_source_phone_from_left_ALOHA_TCP']);P[:3,:3]=P[:3,:3]@np.array([[1,0,0],[0,0,-1],[0,1,0]],float);pinv=np.linalg.inv(P);desired=np.asarray(json.loads((OUT/'target_object_frames.json').read_text())['desired_phone_on_pad']);
 def pose(Rm,p):x=np.eye(4);x[:3,:3]=Rm;x[:3,3]=p;return x
 rec=pose(lr[530],lp[530])@pinv;recn=pose(nlr[530],nlp[530])@pinv
 for label,x in [('EXACT',rec),('NULLSPACE',recn)]:
  dot=np.clip(np.dot(x[:3,1],desired[:3,1]),-1,1);am[f'phone_pad_{label}']={'center_error_mm':float(np.linalg.norm(x[:3,3]-desired[:3,3])*1000),'back_normal_error_deg':float(np.degrees(np.arccos(dot)))}
 dump(OUT/'anchor_metrics.json',{'gate_position_mm':5,'gate_normal_deg':5,'anchors':am})
 status=json.loads((OUT/'ik_metrics.json').read_text())['status'];print(json.dumps({'status':status,'selected_orientation_weight':selected_w,'EXACT':metrics['EXACT'],'NULLSPACE':metrics['NULLSPACE']},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
