#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import *
ROOT=Path('/home/jbnu/aloha_g1_dataset');GEN=ROOT/'outputs/behavior_comparison/generated/episode49_vla_generated_target.npz';OUT=ROOT/'outputs/behavior_comparison/synthetic';XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
CASES=('identical','time_warped','joint_offset','hand_position_offset','left_right_swapped','quaternion_sign_flip','missing_actual_state','different_fps','missing_event')
def resample(a,n):
 x=np.linspace(0,len(a)-1,n);i=np.floor(x).astype(int);j=np.minimum(i+1,len(a)-1);w=x-i
 if np.issubdtype(a.dtype,np.number):return a[i]*(1-w[:,None])+a[j]*w[:,None] if a.ndim>1 else a[i]*(1-w)+a[j]*w
 return a[np.rint(x).astype(int)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,default=GEN);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--case',choices=(*CASES,'all'),default='all');a=p.parse_args();g,gm=load_behavior(a.generated);selected=CASES if a.case=='all' else (a.case,);made=[]
 for case in selected:
  n=len(g['timestamps']);arr={k:v.copy() for k,v in g.items() if k in REQUIRED_ARRAYS or k.startswith('target_') or k in ('valid_frame_mask','state_timeout_mask','interpolation_mask','event_names','event_frames','event_timestamps')};full=g['target_full_qpos'].copy();fps=float(g['fps']);meta=copy.deepcopy(gm);meta.update(schema_version=SCHEMA,behavior_id=f'synthetic_{case}',behavior_role='synthetic_test',source_type=f'synthetic_{case}',source_path=str(a.generated.resolve()),created_at=now(),execution_status='synthetic',is_synthetic=True,valid_for_paper_result=False,notes='DIAGNOSTIC ONLY - NOT A PAPER EXPERIMENT RESULT',synthetic_case=case,expected_result='')
  if case=='time_warped':
   arr['timestamps']=g['timestamps']*1.35;fps=float(g['fps'])/1.35;arr['fps']=np.array(fps);meta.update(fps=fps,duration_sec=float(arr['timestamps'][-1]),expected_result='raw differs; normalized/DTW low')
   if 'event_timestamps' in arr:arr['event_timestamps']=arr['event_timestamps']*1.35
  elif case=='joint_offset':
   joint='left_elbow_joint';k=list(g['full_joint_names']).index(joint);full[:,k]+=.05;meta.update(expected_result='left_elbow_joint RMSE 0.05 rad',synthetic_offset_joint=joint,synthetic_offset_rad=.05)
  elif case=='hand_position_offset':meta.update(expected_result='left hand position RMSE 0.02 m',synthetic_hand_offset_m=[.02,0,0])
  elif case=='left_right_swapped':meta.update(expected_result='comparison refusal',left_right_swapped=True)
  elif case=='quaternion_sign_flip':meta['expected_result']='orientation geodesic error 0'
  elif case=='missing_actual_state':meta['expected_result']='primary result refusal'
  elif case=='different_fps':
   fps=50.;n=int(round((g['timestamps'][-1]-g['timestamps'][0])*fps))+1;full=resample(full,n)
   for k in ('target_full_qpos','target_arm_qpos','target_left_dex3_qpos','target_right_dex3_qpos'):arr[k]=resample(g[k],n)
   for k in ('left_phase','right_phase','valid_frame_mask','state_timeout_mask','interpolation_mask'):arr[k]=resample(g[k],n)
   arr['timestamps']=np.arange(n)/fps;arr['fps']=np.array(fps);meta.update(fps=fps,frame_count=n,duration_sec=float(arr['timestamps'][-1]),expected_result='resampling comparison available')
   if 'event_timestamps' in arr:arr['event_frames']=np.rint(arr['event_timestamps']*fps).astype(np.int64)
  elif case=='missing_event':arr.pop('event_names',None);arr.pop('event_frames',None);arr.pop('event_timestamps',None);meta['expected_result']='phase alignment warning/fallback; other methods available'
  else:meta['expected_result']='all trajectory errors zero'
  if case!='missing_actual_state':
   arr['actual_full_qpos']=full;name=list(g['full_joint_names']);arr['actual_arm_qpos']=full[:,[name.index(x) for x in g['arm_joint_names']]];arr['actual_left_dex3_qpos']=full[:,[name.index(x) for x in g['left_dex3_joint_names']]];arr['actual_right_dex3_qpos']=full[:,[name.index(x) for x in g['right_dex3_joint_names']]];fk=compute_behavior_fk(full,XML);arr.update(fk)
   if case=='hand_position_offset':arr['left_hand_position']=arr['left_hand_position']+np.array([.02,0,0]);arr['bimanual_midpoint']=(arr['left_hand_position']+arr['right_hand_position'])/2;arr['bimanual_relative_position']=arr['right_hand_position']-arr['left_hand_position']
   if case=='quaternion_sign_flip':arr['left_hand_quaternion']*=-1;arr['right_hand_quaternion']*=-1;arr['bimanual_relative_quaternion']*=-1
  else:
   for k in ('left_hand_position','right_hand_position','left_hand_quaternion','right_hand_quaternion','bimanual_midpoint','bimanual_relative_position','bimanual_relative_quaternion'):arr[k]=g[k].copy()
  meta['frame_count']=len(arr['timestamps']);meta['duration_sec']=float(arr['timestamps'][-1]-arr['timestamps'][0]);out=a.output_dir/f'{case}.npz';save_behavior(out,arr,meta);made.append(str(out))
 print(json.dumps({'generated':made,'paper_valid':False},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
