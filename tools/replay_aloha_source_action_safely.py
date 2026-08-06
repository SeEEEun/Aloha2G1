#!/usr/bin/env python3
"""Fail-closed orchestration wrapper. No hardware transport is imported or implemented."""
from __future__ import annotations
import argparse,json,signal,sys,time,uuid
from pathlib import Path
import numpy as np
from aloha_source_validation_common import *
REQUIRED=['enable_hardware_command','confirmed_empty_workspace','confirmed_estop_ready','confirmed_joint_order','confirmed_gripper_direction','operator_present']
def gate(a):
 missing=[f'--{x.replace("_","-")}' for x in REQUIRED if not getattr(a,x)]
 if a.mode=='full' and not a.confirmed_segments_tested:missing.append('--confirmed-segments-tested')
 return missing
def safety_check(command,actual,previous,cfg,age):
 reasons=[]
 if not np.isfinite(command).all() or not np.isfinite(actual).all():reasons.append('NAN_INF')
 if age>cfg['state_stale_seconds']:reasons.append('SUBSCRIBER_TIMEOUT')
 if previous is not None and np.any(abs(command-previous)>cfg['max_step_per_channel']):reasons.append('COMMAND_STEP')
 return reasons
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['inspect','dry-run','read-only-monitor','plan-start','arm','execute']);p.add_argument('--mode',choices=['inspect','dry-run','read-only-monitor','plan-start','segment','full'],default='dry-run');p.add_argument('--start-frame',type=int,default=0);p.add_argument('--end-frame',type=int,default=30);p.add_argument('--speed',type=float,default=.1);p.add_argument('--trial-id',default=None);p.add_argument('--object-configuration',default='none');p.add_argument('--operator',default='UNSPECIFIED');p.add_argument('--video-filename',action='append',default=[])
 for x in REQUIRED+['confirmed_segments_tested']:p.add_argument('--'+x.replace('_','-'),action='store_true')
 a=p.parse_args();q,fps,_=load_action();end=min(a.end_frame,len(q));seg=q[a.start_frame:end];cfg=json.loads((ROOT/'configs/aloha_source_validation_safety.unreviewed.json').read_text());missing=gate(a);trial=a.trial_id or time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6];out=BASE/'trials'/trial;out.mkdir(parents=True,exist_ok=True)
 mapping=BASE/'preflight/joint_mapping_report.json'
 meta={'trial_id':trial,'date_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'operator':a.operator,'robot':'Trossen AI Stationary ALOHA','trajectory_path':str(TRAJECTORY),'trajectory_sha256':sha(TRAJECTORY),'git_commit_hash':'NOT_A_GIT_REPOSITORY','working_tree_dirty':'UNKNOWN_NOT_A_GIT_REPOSITORY','joint_mapping_report_hash':sha(mapping) if mapping.exists() else None,'playback_speed':a.speed,'frame_range':[a.start_frame,end-1],'object_configuration':a.object_configuration,'phone_present':'phone' in a.object_configuration,'accessory_present':'accessory' in a.object_configuration,'charger_present':'charger' in a.object_configuration,'emergency_stop_confirmed':a.confirmed_estop_ready,'empty_workspace_confirmed':a.confirmed_empty_workspace,'gripper_direction_verified':a.confirmed_gripper_direction,'start_state_error':'NOT_AVAILABLE_DRY_RUN','completion_status':'NOT_EXECUTED','stop_reason':None,'video_filenames':a.video_filename,'human_annotations':[],'hardware_transport':'NOT_IMPLEMENTED_FAIL_CLOSED'}
 dump(out/'trial_metadata.json',meta);dump(out/'video_manifest.json',{'filenames':a.video_filename,'recording':'EXTERNAL_MANUAL'})
 if a.stage in ('inspect','dry-run','read-only-monitor','plan-start'):
  np.savez_compressed(out/'command_action.npz',timestamp_command=np.arange(len(seg))/(fps*a.speed),frame_index=np.arange(a.start_frame,end),command_action_14d=seg,left_gripper_command=seg[:,6],right_gripper_command=seg[:,13],segment_id=np.asarray(a.mode),playback_speed=a.speed,data_provenance=np.asarray('DRY_RUN_NO_PUBLISHER'))
  meta['completion_status']='DRY_RUN_PASS';dump(out/'trial_metadata.json',meta);print(json.dumps({'status':'DRY_RUN_PASS','publisher_created':False,'command_client_created':False,'frames':len(seg),'trial':trial},indent=2));return 0
 if missing:raise RuntimeError('HARDWARE COMMAND BLOCKED: missing gates '+', '.join(missing))
 if a.stage=='arm':raise RuntimeError('ARM TOKEN REFUSED: no verified non-mutating live state backend and no reviewed hardware limits/mapping artifact')
 raise RuntimeError('EXECUTE REFUSED: hardware transport intentionally not implemented; existing connect() moves robot and violates read-only preflight')
if __name__=='__main__':raise SystemExit(main())
