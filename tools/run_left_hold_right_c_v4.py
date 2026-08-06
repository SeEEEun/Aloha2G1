#!/usr/bin/env python3
"""Fail-closed left A+B scene-registration gate before portrait/right-C work."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/left_hold_v4_mpl')
import matplotlib.pyplot as plt
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_left_hold_right_c_v4'
SRC=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
GRASP=ROOT/'converted_runs/g1_left_phone_cgap_grasp/selected_left_phone_cgap_grasp.npz'
GREPORT=GRASP.with_name('selected_left_phone_cgap_grasp_report.json')
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
import retarget_episode49_optimized_action_to_g1 as epi
PHONE=np.array([.525,.07,.83075]);ROOTPOS=np.array([.44514890950197095,-.35257022755443246,.7922728583]);ROOTQ=np.array([.7071067812,0,0,.7071067812])
EVENTS=('left_phone_grasp_start','phone_rotation_to_portrait_start','phone_portrait_reached','right_accessory_grasp_start','accessory_detachment_start','accessory_removed','phone_move_to_charger_start','phone_charger_attachment_complete','left_phone_release_complete','left_arm_return_near_home','task_end')
def ser(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:ser(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [ser(v) for v in x]
 return x
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(ser(x),indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);tl=json.loads((ROOT/'configs/episode49_task_timeline.approved.json').read_text());ev={x['event']:x['frame'] for x in tl['events']};assert all(x in ev for x in EVENTS)
 with np.load(SRC) as z:act=z['optimized_action'].astype(float);ts=z['timestamp'].astype(float);assert act.shape==(990,14) and np.isfinite(act).all() and float(z['fps'])==30
 am,_=epi.aloha.load_validated_model(epi.ALOHA_XML);aq,_=epi.aloha.mapped_qpos(act);fk=epi.aloha.fk(am,aq)
 frames=np.array([ev[x] for x in EVENTS]); lp=fk['left_position_m'];lr=fk['left_quaternion_wxyz'];rp=fk['right_position_m'];rr=fk['right_quaternion_wxyz']
 np.savez_compressed(OUT/'aloha_portrait_motion_prior.npz',event_names=np.array(EVENTS),event_frames=frames,timestamp=ts,left_position=lp,left_quaternion_wxyz=lr,right_position=rp,right_quaternion_wxyz=rr,left_relative_translation_from_grasp=lp-lp[ev['left_phone_grasp_start']],optimized_action=act,fps=30.)
 model=mujoco.MjModel.from_xml_path(str(epi.G1_XML));data=mujoco.MjData(model);info=epi.ik.validate_model(epi.G1_XML);wid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_wrist_yaw_link')
 with np.load(GRASP) as z:qsel=z['full_g1_qpos'].copy();oldphone=z['phone_proxy_pose'][:3];contact=z['actual_contact_points'];normals=z['actual_contact_normals'];hand=z['left_dex3_qpos'];base_arm=z['left_arm_qpos']
 data.qpos[:]=qsel;mujoco.mj_forward(model,data);wp=data.xpos[wid].copy();wR=data.xmat[wid].reshape(3,3).copy();relp=wp-oldphone;relR=wR
 qbase=qsel.copy();qbase[:3]=ROOTPOS;qbase[3:7]=ROOTQ;ids=np.asarray(info['arm_qpos_ids'][:7]);lo=info['joint_limits'][:7,0]+1e-4;hi=info['joint_limits'][:7,1]-1e-4
 seeds=[base_arm.copy(),info['stand_arm_q'][:7].copy()]
 for j in range(8):
  q=base_arm.copy();k=[1,2,3,4,5,6,0,3][j];q[k]+=[-.4,.4,-.35,.35,-.3,.3,.25,-.25][j];seeds.append(np.clip(q,lo,hi))
 rows=[]
 for assignment,flip in [('A_SCREEN_B_BACK',False),('A_BACK_B_SCREEN',True)]:
  F=Rotation.from_rotvec([0,0,np.pi]).as_matrix() if flip else np.eye(3);tR=F@relR;tp=PHONE+F@relp
  for si,x0 in enumerate(seeds):
   def fun(x):
    data.qpos[:]=qbase;data.qpos[ids]=x;mujoco.mj_forward(model,data);p=data.xpos[wid];R=data.xmat[wid].reshape(3,3);rv=Rotation.from_matrix(R.T@tR).as_rotvec();return np.r_[45*(p-tp),2.5*rv,.015*(x-x0)]
   sol=least_squares(fun,np.clip(x0,lo,hi),bounds=(lo,hi),max_nfev=650,ftol=1e-10,xtol=1e-10,gtol=1e-10);data.qpos[:]=qbase;data.qpos[ids]=sol.x;mujoco.mj_forward(model,data);errvec=data.xpos[wid]-tp;ang=np.degrees(Rotation.from_matrix(data.xmat[wid].reshape(3,3).T@tR).magnitude());margin=np.minimum(sol.x-info['joint_limits'][:7,0],info['joint_limits'][:7,1]-sol.x)
   rows.append({'assignment':assignment,'seed':si,'optimizer_success':bool(sol.success),'nfev':sol.nfev,'target_error_vector_xyz_m':errvec,'position_error_m':np.linalg.norm(errvec),'orientation_error_deg':ang,'joint_margin_rad':margin,'minimum_joint_margin_rad':margin.min(),'wrist_target_world':tp,'q':sol.x,'carrier_gate_pass':bool(np.linalg.norm(errvec)<=.003 and ang<=10 and margin.min()>=0)})
 best=min(rows,key=lambda x:(not x['carrier_gate_pass'],x['position_error_m'],x['orientation_error_deg']))
 public=[{k:ser(v) for k,v in x.items() if k!='q'} for x in rows];dump('left_ab_candidate_results.json',{'candidate_count':len(rows),'assignment_count':2,'source_local_grasp_report':str(GREPORT),'source_local_grasp_sha256':sha(GRASP),'candidates':public})
 passed=[x for x in rows if x['carrier_gate_pass']]
 if not passed:
  status='BLOCKED_LEFT_AB_PHONE_GRASP';sel={'status':status,'selected_left_ab_grasp':None,'best_failed':{k:ser(v) for k,v in best.items() if k!='q'},'reason':'Validated local A+B grasp carrier transform cannot be registered to the immutable current scene phone with <=3 mm and <=10 deg wrist-pose error. Fingertip collision/contact acceptance was therefore not claimed.'}
 else: status='LEFT_AB_CARRIER_READY_NEEDS_CONTACT_REVALIDATION';sel={'status':status,'selected_left_ab_grasp':{k:ser(v) for k,v in passed[0].items() if k!='q'}}
 dump('selected_left_ab_grasp.json',sel);dump('phone_grasp_transform.json',{'status':'SOURCE_LOCAL_VALIDATED_BUT_CURRENT_SCENE_REGISTRATION_FAILED' if not passed else 'CANDIDATE','T_phone_from_left_wrist_translation_m':relp,'R_phone_from_left_wrist_source':relR.T,'T_AB_pinch_from_phone':'NOT_ESTABLISHED_IN_CURRENT_SCENE' if not passed else 'PENDING_CONTACT_REVALIDATION'})
 blocked={'status':'NOT_GENERATED_LEFT_AB_GATE_FAILED','source_derived_pose':None,'corrections_invented':False};dump('portrait_hold_pose_candidates.json',blocked);dump('selected_portrait_hold_pose.json',blocked);dump('right_c_axial_candidates.json',{'status':'NOT_RUN_LEFT_AB_GATE_FAILED','candidate_count':0});dump('right_c_radial_gap_candidates.json',{'status':'NOT_RUN_LEFT_AB_GATE_FAILED','candidate_count':0});dump('selected_right_c_candidate.json',{'status':'NOT_RUN_LEFT_AB_GATE_FAILED','selected_candidate':None});dump('coupled_feasibility_metrics.json',{'status':status,'full_990_frame_v3_resume':False,'aloha_event_order_unchanged':True});dump('collision_breakdown.json',{'status':'NOT_EVALUATED_CARRIER_IK_FAILED','best_failed_target_error_vector_xyz_m':best['target_error_vector_xyz_m'],'joint_margins_rad':best['joint_margin_rad'],'joint_margin_names':['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint'],'diagnostic_phone_translation_that_would_reduce_carrier_residual_m':best['target_error_vector_xyz_m'],'translation_applied':False,'translation_note':'Diagnostic sensitivity only; immutable scene phone coordinates were not changed.','phone_ring_palm_clearance_m':None,'forbidden_pairs':[]})
 fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(rows));ax.bar(x,[r['position_error_m']*1000 for r in rows]);ax.axhline(3,color='r',label='3 mm gate');ax.set(xlabel='assignment/seed candidate',ylabel='wrist carrier position error [mm]',title=f'Left A+B current-scene registration | {status}');ax.legend();fig.tight_layout();fig.savefig(OUT/'left_ab_grasp_front.png',dpi=170);fig.savefig(OUT/'left_ab_grasp_side.png',dpi=170);plt.close(fig)
 for n,title in [('portrait_hold_front.png','NOT GENERATED: LEFT A+B GATE FAILED'),('portrait_hold_side.png','NOT GENERATED: LEFT A+B GATE FAILED'),('portrait_hold_top.png','NOT GENERATED: LEFT A+B GATE FAILED'),('right_c_axial_best.png','NOT RUN: PORTRAIT POSE UNAVAILABLE'),('right_c_gap_best.png','NOT RUN: PORTRAIT POSE UNAVAILABLE'),('combined_left_hold_right_preinsert.png','NOT RUN: LEFT A+B GATE FAILED')]:
  fig,ax=plt.subplots(figsize=(8,4));ax.text(.5,.5,title,ha='center',va='center',fontsize=14);ax.axis('off');fig.savefig(OUT/n,dpi=150);plt.close(fig)
 report=f'''# Episode 49 left-hold/right-C v4\n\nStatus: `{status}`\n\nThe earlier right-C failure is pose-conditioned, not proof that C is unusable. The dependency-ordered retry stopped at left A+B current-scene registration. Twenty deterministic carrier-IK candidates covered both screen/back assignments. Valid: {len(passed)}. Best failed position error: {best['position_error_m']*1000:.6f} mm; orientation error: {best['orientation_error_deg']:.6f} deg. No portrait pose, right-C candidate, continuous insertion, or 990-frame trajectory was generated.\n\nSIMULATION ONLY — NO REAL ROBOT COMMANDS — NO DDS OR PUBLISHER\nALOHA SOURCE MOTION PRESERVED AS THE RETARGETING PRIOR\nPHONE PORTRAIT POSE MUST BE DERIVED BEFORE RIGHT-C INSERTION\nREAL G1 SAFETY NOT_PERFORMED\n''';(OUT/'report.md').write_text(report);(OUT/'report').mkdir(exist_ok=True);(OUT/'report/index.html').write_text(f'<h1>{status}</h1><p>No portrait/right-C/full trajectory was generated.</p>')
 print(status);print('BEST_ERROR_MM',best['position_error_m']*1000);print('BEST_ORIENTATION_DEG',best['orientation_error_deg']);return 0 if status.startswith('LEFT_AB') else 2
if __name__=='__main__':raise SystemExit(main())
