#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from aloha_source_validation_common import *
CFG=ROOT/'configs/aloha_source_validation_safety.unreviewed.json'
def main():
 p=argparse.ArgumentParser();p.add_argument('--state-record',type=Path,required=True);p.add_argument('--mapping-report',type=Path,default=BASE/'preflight/joint_mapping_report.json');p.add_argument('--config',type=Path,default=CFG);p.add_argument('--output',type=Path,default=BASE/'preflight');a=p.parse_args()
 d=load_record(a.state_record);q,_,_=load_action();cur=np.asarray(d['mapped_observation_state_14d'][-1],float);dif=cur-q[0];cfg=json.loads(a.config.read_text());thr=np.asarray(cfg['start_error_threshold_per_channel']);mapping=json.loads(a.mapping_report.read_text()) if a.mapping_report.exists() else {'status':'MISSING'};finite=np.isfinite(cur).all();stale=(time.time()-float(d['timestamp_wall'][-1]))>cfg['state_stale_seconds'];grip_verified=False
 blockers=[]
 if not finite:blockers.append('NaN/Inf')
 if mapping.get('status')!='PASS':blockers.append('joint order unverified')
 if np.any(np.abs(dif)>thr):blockers.append('start error threshold exceeded')
 if not grip_verified:blockers.append('live gripper direction unverified')
 if stale and str(d.get('data_provenance',''))!='SIM_FIXTURE_SYNTHETIC_NOT_REAL':blockers.append('stale state')
 rows=[{'index':i,'name':NAMES[i],'current':cur[i],'action0':q[0,i],'signed_difference':dif[i],'absolute_difference':abs(dif[i]),'threshold':thr[i],'pass':bool(abs(dif[i])<=thr[i])} for i in range(14)];a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/'start_state_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 r={'status':'BLOCK' if blockers else 'PASS','frame0_direct_command_allowed':False,'blockers':blockers,'threshold_source':str(a.config),'threshold_status':cfg['status'],'units':cfg['units'],'current_state_14d':cur.tolist(),'optimized_action_0':q[0].tolist(),'signed_difference':dif.tolist(),'absolute_difference':np.abs(dif).tolist(),'l2_norm':float(np.linalg.norm(dif)),'group_max_abs':{'left_arm':float(np.max(abs(dif[:6]))),'left_gripper':float(abs(dif[6])),'right_arm':float(np.max(abs(dif[7:13]))),'right_gripper':float(abs(dif[13]))},'state_provenance':str(d.get('data_provenance','UNKNOWN'))}
 dump(a.output/'start_state_comparison.json',r);print(json.dumps(r,indent=2));return 0 if r['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
