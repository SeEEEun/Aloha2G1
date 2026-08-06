#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import numpy as np
from aloha_source_validation_common import *
CFG=ROOT/'configs/aloha_source_validation_safety.unreviewed.json'
def main():
 p=argparse.ArgumentParser();p.add_argument('--state-record',type=Path,required=True);p.add_argument('--mapping-report',type=Path,default=BASE/'preflight/joint_mapping_report.json');p.add_argument('--duration',type=float,default=5);p.add_argument('--fps',type=float,default=30);p.add_argument('--output',type=Path,default=BASE/'start_transition');a=p.parse_args();cfg=json.loads(CFG.read_text());mapping=json.loads(a.mapping_report.read_text());d=load_record(a.state_record);start=np.asarray(d['mapped_observation_state_14d'][-1],float);target=load_action()[0][0].copy();target[[6,13]]=start[[6,13]]
 n=max(3,int(a.duration*a.fps)+1);u=np.linspace(0,1,n);s=10*u**3-15*u**4+6*u**5;plan=start+(target-start)*s[:,None];step=np.diff(plan,axis=0);vel=step*a.fps;acc=np.diff(plan,n=2,axis=0)*a.fps*a.fps;checks={'finite':bool(np.isfinite(plan).all()),'mapping':mapping.get('status')=='PASS','max_step':bool(np.all(np.max(abs(step),axis=0)<=cfg['max_step_per_channel'])),'max_velocity':bool(np.all(np.max(abs(vel),axis=0)<=cfg['max_velocity_per_channel'])),'max_acceleration':bool(np.all(np.max(abs(acc),axis=0)<=cfg['max_acceleration_per_channel'])),'joint_range':'BLOCKED_NO_VERIFIED_HARDWARE_LIMITS'};passed=all(v is True for v in checks.values())
 a.output.mkdir(parents=True,exist_ok=True)
 if passed:np.savez_compressed(a.output/'planned_start_transition.npz',action=plan,fps=a.fps,joint_names=NAMES,data_provenance=np.asarray('DRY_RUN_PLAN_NOT_EXECUTABLE'))
 metrics={'status':'PASS' if passed else 'BLOCK','execution_allowed':False,'checks':checks,'gripper_policy':'HOLD_CURRENT','duration_s':a.duration,'frames':n,'max_step_per_channel':np.max(abs(step),axis=0).tolist(),'max_velocity_per_channel':np.max(abs(vel),axis=0).tolist(),'max_acceleration_per_channel':np.max(abs(acc),axis=0).tolist(),'threshold_source':str(CFG)};dump(a.output/'transition_metrics.json',metrics);(a.output/'validation_report.md').write_text('# Start transition validation\n\n'+json.dumps(metrics,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(12,5));ax.plot(np.arange(n)/a.fps,plan);fig.savefig(a.output/'per_joint_plot.png',dpi=140);plt.close(fig);print(json.dumps(metrics,indent=2));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
