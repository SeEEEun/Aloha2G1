#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import numpy as np
from aloha_source_validation_common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=TRAJECTORY);p.add_argument('--output',type=Path,default=BASE/'inspect');a=p.parse_args()
 try:q,fps,keys=load_action(a.input)
 except Exception as e:print(f'FAIL: {e}',file=sys.stderr);return 2
 a.output.mkdir(parents=True,exist_ok=True);step=np.diff(q,axis=0);vel=step*fps;acc=np.diff(q,n=2,axis=0)*fps*fps
 rows=[]
 for i,n in enumerate(NAMES):rows.append({'channel':i,'name':n,'semantic':SEMANTICS[i],'min':q[:,i].min(),'max':q[:,i].max(),'mean':q[:,i].mean(),'std':q[:,i].std(),'max_abs_step':np.abs(step[:,i]).max(),'max_abs_velocity':np.abs(vel[:,i]).max(),'max_abs_acceleration':np.abs(acc[:,i]).max()})
 for fn,data in [('action_channel_stats.csv',rows)]:
  with (a.output/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 with (a.output/'action_step_stats.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['frame_from','frame_to','max_abs_step','channel']);
  for i,r in enumerate(np.abs(step)):w.writerow([i,i+1,float(r.max()),int(r.argmax())])
 flat=np.unravel_index(np.abs(step).argmax(),step.shape)
 events=[]
 for side,ch in [('left',6),('right',13)]:
  d=np.diff(q[:,ch]);threshold=max(float(np.percentile(np.abs(d),95)),1e-6)
  for i in np.flatnonzero(np.abs(d)>=threshold):events.append({'side':side,'frame':int(i+1),'delta':float(d[i]),'candidate':'OPEN' if d[i]>0 else 'CLOSE','evidence':'increasing aperture convention; temporal signal derivative','review_status':'REQUIRES_HUMAN_REVIEW'})
 dump(a.output/'gripper_events.json',{'direction_evidence':'increasing=open from stationary ALOHA aperture model and map_aloha_gripper_to_dex3.py; hardware command/state sign not independently live-verified','events':events})
 t=np.arange(len(q))/fps
 fig,ax=plt.subplots();ax.plot(t,q[:,6],label='left');ax.plot(t,q[:,13],label='right');ax.legend();ax.set(xlabel='s',ylabel='native gripper position');fig.savefig(a.output/'gripper_signals.png',dpi=140);plt.close(fig)
 for fn,data,title in [('joint_velocity.png',vel,'Approx velocity'),('joint_acceleration.png',acc,'Approx acceleration')]:
  fig,ax=plt.subplots(figsize=(12,5));ax.plot(np.arange(len(data))/fps,data);ax.set(title=title,xlabel='s');fig.savefig(a.output/fn,dpi=140);plt.close(fig)
 summary={'status':'PASS','input':str(a.input.resolve()),'sha256':sha(a.input),'npz_keys':keys,'action_key':'optimized_action','offline_temporal_consensus':True,'online_closed_loop':False,'shape':list(q.shape),'dtype_original':str(np.load(a.input)['optimized_action'].dtype),'finite':True,'fps':fps,'duration_s':len(q)/fps,'channel_order':NAMES,'gripper_indices':[6,13],'max_change':{'from_frame':int(flat[0]),'to_frame':int(flat[0]+1),'channel':int(flat[1]),'name':NAMES[flat[1]],'signed_step':float(step[flat])},'stats':rows}
 dump(a.output/'action_summary.json',summary);print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
