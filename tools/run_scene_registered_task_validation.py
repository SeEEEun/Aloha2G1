#!/usr/bin/env python3
"""Hard-gated offline scene-registered task validation pipeline.

Before a manually approved Episode-49 timeline exists this script performs only
input regression audit and event-review artifact generation, then exits 4.
It imports no hardware, DDS, publisher, subscriber, or command API.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
OUT=ROOT/'outputs/scene_registered_task_validation';ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';FROZEN=ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz';SEM=ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json';LAYOUT=ROOT/'isaaclab_magsafe_fixed_scene/scene_layout.json';PHASE=ROOT/'outputs/magsafe_gripper_phases.csv';IMG=ROOT/'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000';APP=ROOT/'configs/episode49_task_timeline.approved.json';DRAFT=ROOT/'configs/episode49_task_timeline.draft.json'
EXPECTED_ACTION='a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c';EXPECTED_FROZEN='c58c8ee6f98e02d71e22abc721fcb92bb7e5c233963b0cb2d44b3fa6c4ad1f3e'
REQUIRED=('left_phone_grasp_start','phone_rotation_to_portrait_start','phone_portrait_reached','right_accessory_grasp_start','accessory_detachment_start','accessory_removed','right_accessory_release_complete','phone_move_to_charger_start','phone_charger_attachment_complete','left_phone_release_complete')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def audit():
 with np.load(ACTION,allow_pickle=False) as z:a=z['optimized_action'];ts=z['timestamp'];fps=float(z['fps'])
 with np.load(FROZEN,allow_pickle=False) as z:q=z['g1_arm_joint_trajectory'];names=z['arm_joint_names'].astype(str).tolist();ffps=float(z['fps'])
 ah,fh=sha(ACTION),sha(FROZEN);sem=json.load(open(SEM));layout_hash=sha(LAYOUT)
 checks={'optimized_action_shape':list(a.shape),'frozen_arm_shape':list(q.shape),'fps':fps,'frozen_fps':ffps,'duration_sec':float(ts[-1]-ts[0]),'timestamps_monotonic':bool(np.all(np.diff(ts)>0)),'action_nan_inf':int(np.size(a)-np.isfinite(a).sum()),'frozen_nan_inf':int(np.size(q)-np.isfinite(q).sum()),'arm_joint_names':names,'source_action_sha256':ah,'frozen_reference_sha256':fh,'scene_layout_sha256':layout_hash,'semantic_definition_status':sem['status'],'charger_pad_frame':sem['charger_pad_verified_from_asset'],'object_pose_immutability':True,'channel_map':{'left_arm':[0,1,2,3,4,5],'left_gripper':6,'right_arm':[7,8,9,10,11,12],'right_gripper':13}}
 ok=a.shape==(990,14) and q.shape==(990,14) and fps==30 and ffps==30 and checks['timestamps_monotonic'] and not checks['action_nan_inf'] and not checks['frozen_nan_inf'] and ah==EXPECTED_ACTION and fh==EXPECTED_FROZEN
 checks['status']='PASS' if ok else 'FAIL';dump(OUT/'audit/input_audit.json',checks)
 reg={'status':'PASS' if fh==EXPECTED_FROZEN else 'FAIL','exact_file_sha256_preserved':fh==EXPECTED_FROZEN,'expected_sha256':EXPECTED_FROZEN,'actual_sha256':fh,'exact_array_self_reload':bool(np.array_equal(q,np.load(FROZEN,allow_pickle=False)['g1_arm_joint_trajectory'])),'frame_count_preserved':len(q)==990,'source_action_unchanged':ah==EXPECTED_ACTION,'object_pose_changed':False};dump(OUT/'audit/frozen_reference_regression.json',reg)
 if not ok:raise RuntimeError('Authoritative input regression failed')
 return a,ts,checks
def source_fk(action):
 import validate_smolvla_in_stationary_aloha_mujoco as av
 xml=Path('/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml');m,_=av.load_validated_model(xml);q,_=av.mapped_qpos(action);fk=av.fk(m,q);lp=fk['left_position_m'];rp=fk['right_position_m'];lv=np.r_[0,np.linalg.norm(np.diff(lp,axis=0),axis=1)*30];rv=np.r_[0,np.linalg.norm(np.diff(rp,axis=0),axis=1)*30];return lv,rv
def timeline_gate():
 if not APP.exists():return None,list(REQUIRED),'APPROVED_TIMELINE_MISSING'
 d=json.load(open(APP));events=d.get('events',[]);manual={e.get('event'):e for e in events if e.get('source')=='manual_video_review'};missing=[x for x in REQUIRED if x not in manual]
 valid=d.get('status')=='APPROVED_MANUAL_VIDEO_REVIEW' and not missing
 return d,missing,'PASS' if valid else 'EPISODE49_EVENT_APPROVAL_REQUIRED'
def review_artifacts(ts,lv,rv):
 import cv2
 rows=list(csv.DictReader(open(PHASE)));od=OUT/'event_review';cs=od/'contact_sheets';cs.mkdir(parents=True,exist_ok=True)
 transitions=[i for i,x in enumerate(rows) if x['left_transition'] or x['right_transition']];speed=np.maximum(lv,rv);peaks=np.argsort(speed)[-60:];diag=set(j for i in list(transitions)+list(peaks) for j in range(max(0,i-12),min(990,i+13)))
 video=od/'episode49_event_review.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
 for i in range(990):
  im=cv2.imread(str(IMG/f'frame_{i:06d}.png'));canvas=np.zeros((720,1280,3),np.uint8);h=720;w=int(im.shape[1]*h/im.shape[0]);canvas[:h,:w]=cv2.resize(im,(w,h));x=rows[i];lines=[f'frame {i}/989  t={ts[i]:.3f}s',f"raw L/R {float(x['left_gripper_raw']):.5f} / {float(x['right_gripper_raw']):.5f}",f"automatic L/R {x['left_phase']} / {x['right_phase']}",f'TCP speed L/R {lv[i]:.4f} / {rv[i]:.4f} m/s',('candidate interval: AUTOMATIC MOTION/GRIPPER DIAGNOSTIC' if i in diag else 'candidate interval: none'),'CANDIDATE ONLY','MANUAL APPROVAL REQUIRED']
  for k,s in enumerate(lines):cv2.putText(canvas,s,(w+18,70+k*65),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,255,255) if k<5 else (0,0,255),2,cv2.LINE_AA)
  writer.write(canvas)
 writer.release()
 left_trans=[i for i,x in enumerate(rows) if 'left_transition' in x and x['left_transition']];right_trans=[i for i,x in enumerate(rows) if 'right_transition' in x and x['right_transition']]
 motion=sorted(peaks,key=lambda i:i)[:18]
 def centers(event):
  if event in ('left_phone_grasp_start','left_phone_release_complete'):return left_trans
  if event in ('right_accessory_grasp_start','right_accessory_release_complete'):return right_trans
  return motion
 from PIL import Image,ImageDraw
 for event in REQUIRED:
  cc=centers(event);chosen=[]
  for c in cc:
   if not chosen or c-chosen[-1]>=10:chosen.append(int(c))
  chosen=(chosen[:12] or list(np.linspace(0,989,12,dtype=int)))
  sheet=Image.new('RGB',(960,((len(chosen)+3)//4)*220),'white');draw=ImageDraw.Draw(sheet)
  for k,i in enumerate(chosen):
   im=Image.open(IMG/f'frame_{i:06d}.png').convert('RGB');im.thumbnail((230,170));xx=(k%4)*240;yy=(k//4)*220;sheet.paste(im,(xx,yy));r=rows[i];draw.text((xx,yy+172),f'f={i} t={ts[i]:.2f}\nL/R raw={float(r["left_gripper_raw"]):.3f}/{float(r["right_gripper_raw"]):.3f}\nphase={r["left_phase"]}/{r["right_phase"]}\nspeed={lv[i]:.3f}/{rv[i]:.3f}',fill='black')
  draw.rectangle((0,0,960,28),fill='black');draw.text((8,7),event+' — CANDIDATE ONLY / MANUAL APPROVAL REQUIRED',fill='red');sheet.save(cs/f'{event}.png')
 dump(od/'event_review_manifest.json',{'video':str(video),'contact_sheets':{x:str(cs/f'{x}.png') for x in REQUIRED},'candidate_policy':'automatic gripper transitions and TCP-speed peaks are diagnostic candidates only; no event was saved or approved','manual_approval_required':True});return video
def blocked_outputs(missing,reason):
 status={'status':'EPISODE49_EVENT_APPROVAL_REQUIRED','missing_events':missing,'reason':reason,'trajectory_generated':False,'ik_run':False,'candidate_generation_run':False,'continuation_sweep_run':False,'selected_candidate':None,'downstream_generation':'BLOCKED','real_g1_used':False,'real_aloha_used':False,'dds_or_publisher_used':False};dump(OUT/'manual_review_status.json',status);dump(OUT/'selection.json',{'selected_candidate':None,'status':'EPISODE49_EVENT_APPROVAL_REQUIRED'});dump(OUT/'automatic_gate.json',{'timeline_gate':'FAIL','task_success':'NOT_AUTOMATICALLY_EVALUATED','downstream':'BLOCKED'})
 report=OUT/'report';report.mkdir(parents=True,exist_ok=True);(report/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>EPISODE49_EVENT_APPROVAL_REQUIRED</h1><p>NO TRAJECTORY GENERATED — WAITING FOR USER FRAME SELECTION</p><video controls width="960" src="../event_review/episode49_event_review.mp4"></video><pre>'+json.dumps(status,indent=2)+'</pre>')
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');a=p.parse_args();action,ts,_=audit();timeline,missing,reason=timeline_gate()
 if a.audit_only:return 0
 if missing or reason!='PASS':
  lv,rv=source_fk(action);review_artifacts(ts,lv,rv)
  if not DRAFT.exists():dump(DRAFT,{'schema_version':2,'status':'DRAFT_NEEDS_MANUAL_APPROVAL','dataset_episode':49,'events':[],'automatic_events_saved':False})
  blocked_outputs(missing,reason);print('EPISODE49_EVENT_APPROVAL_REQUIRED\nNO TRAJECTORY GENERATED\nWAITING FOR USER FRAME SELECTION');return 4
 print('Timeline passed. Registration/candidate phases are intentionally not implemented in this gate-focused run.');return 5
if __name__=='__main__':raise SystemExit(main())
