#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import *
ROOT=Path('/home/jbnu/aloha_g1_dataset');SRC=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';OUT=ROOT/'outputs/behavior_comparison/generated/episode49_vla_generated_target.npz';XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=SRC);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--xml',type=Path,default=XML);a=p.parse_args()
 with np.load(a.input,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 arr={k:d[k] for k in ('timestamps','fps','full_joint_names','arm_joint_names','left_dex3_joint_names','right_dex3_joint_names','left_phase','right_phase')};arr.update(target_full_qpos=d['full_qpos'],target_arm_qpos=d['arm_qpos'],target_left_dex3_qpos=d['left_dex3_qpos'],target_right_dex3_qpos=d['right_dex3_qpos'],valid_frame_mask=np.ones(len(d['timestamps']),bool),state_timeout_mask=np.zeros(len(d['timestamps']),bool),interpolation_mask=np.zeros(len(d['timestamps']),bool));arr.update(compute_behavior_fk(arr['target_full_qpos'],a.xml));fps=float(d['fps']);q=arr['target_arm_qpos'];arr['joint_velocity']=np.gradient(q,1/fps,axis=0);arr['joint_acceleration']=np.gradient(arr['joint_velocity'],1/fps,axis=0)
 events=[]
 for side in ('left','right'):
  ph=arr[f'{side}_phase'];
  for i in np.r_[0,np.flatnonzero(ph[1:]!=ph[:-1])+1]:events.append((f'{side}_{str(ph[i]).lower()}_start',int(i)))
 events.extend((('task_start',0),('task_end',len(q)-1)));events=sorted(events,key=lambda x:(x[1],x[0]));arr['event_names']=np.asarray([x[0] for x in events]);arr['event_frames']=np.asarray([x[1] for x in events]);arr['event_timestamps']=arr['timestamps'][arr['event_frames']]
 meta={'schema_version':SCHEMA,'behavior_id':'episode49_vla_generated_target','behavior_role':'generated_target','task_name':'magsafe_phone_accessory','robot':'Unitree G1','hand':'Dex3','source_type':'smolvla_temporal_consensus_relative_task_space_transfer','source_path':str(a.input.resolve()),'created_at':now(),'fps':fps,'frame_count':len(q),'duration_sec':float(arr['timestamps'][-1]-arr['timestamps'][0]),'coordinate_frame':'g1_base_fixed_mujoco_world_frame','joint_units':'radian','orientation_representation':'quaternion_wxyz','execution_status':'not_executed','is_synthetic':False,'valid_for_paper_result':True,'paper_result_scope':'target-level generated behavior only','primitive_source':str(d['primitive_source']),'authoritative_for_real_robot':False,'fk_bodies':['left_wrist_yaw_link','right_wrist_yaw_link'],'notes':'Frozen generated target; contains no actual robot state.'}
 save_behavior(a.output,arr,meta);print(json.dumps({'output':str(a.output),'summary':summarize_behavior(arr,meta),'events':len(events)},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
