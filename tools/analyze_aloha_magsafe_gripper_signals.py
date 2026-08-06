#!/usr/bin/env python3
"""Analyze the authoritative optimized ALOHA gripper channels offline."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from magsafe_gripper_common import *

OUT=Path("/home/jbnu/aloha_g1_dataset/outputs/gripper_phase_analysis")
def cli():
 p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=DEFAULT_ACTION); p.add_argument("--output-dir",type=Path,default=OUT); p.add_argument("--fps",type=float); p.add_argument("--left-gripper-index",type=int,default=LEFT_INDEX); p.add_argument("--right-gripper-index",type=int,default=RIGHT_INDEX); p.add_argument("--inspect-only",action="store_true"); return p.parse_args()
def side_stats(raw,filt,fps):
 lo,hi,threshold=two_cluster_threshold(filt); span=hi-lo; derivative=np.gradient(filt)*fps
 # Established ALOHA qpos semantics: aperture increases toward OPEN.
 close=filt < lo+.35*span; opened=filt > lo+.65*span
 events=[]
 for label,mask in (("close_threshold",close),("open_threshold",opened)):
  for i in np.flatnonzero(mask[1:]!=mask[:-1])+1: events.append({"frame":int(i),"event":label+('_enter' if mask[i] else '_exit')})
 unique=np.unique(np.round(raw,8))
 events.sort(key=lambda e:e['frame'])
 semantic={"close_start_frames":[e['frame'] for e in events if e['event']=='open_threshold_exit'],"close_stabilized_frames":[e['frame'] for e in events if e['event']=='close_threshold_enter'],"reopen_start_frames":[e['frame'] for e in events if e['event']=='close_threshold_exit'],"reopen_stabilized_frames":[e['frame'] for e in events if e['event']=='open_threshold_enter']}
 return {"min":float(raw.min()),"max":float(raw.max()),"median":float(np.median(raw)),"percentiles":{str(p):float(np.percentile(raw,p)) for p in (1,5,25,75,95,99)},"initial":float(raw[0]),"final":float(raw[-1]),"noise_mad_diff":float(np.median(np.abs(np.diff(raw)-np.median(np.diff(raw))))),"direction":"increasing_is_open","open_value_estimate":hi,"close_value_estimate":lo,"threshold":threshold,"open_threshold":lo+.65*span,"close_threshold":lo+.35*span,"binary_or_continuous":"binary" if len(unique)<=4 else "continuous","normalization":"physical_aperture_m_not_normalized","saturation":{"near_min_fraction":float(np.mean(raw<=raw.min()+.01*max(np.ptp(raw),1e-12))),"near_max_fraction":float(np.mean(raw>=raw.max()-.01*max(np.ptp(raw),1e-12)))},"nan_inf_count":int((~np.isfinite(raw)).sum()),"events":events,"semantic_event_candidates":semantic,"derivative":derivative}
def main():
 a=cli(); action,t,fps=load_action(a.input,a.left_gripper_index,a.right_gripper_index,a.fps); raw={"left":action[:,a.left_gripper_index],"right":action[:,a.right_gripper_index]}; filt={s:smooth(x) for s,x in raw.items()}; stats={s:side_stats(raw[s],filt[s],fps) for s in raw}
 summary={"input":str(a.input.resolve()),"array_key":DEFAULT_ACTION_KEY,"action_shape":list(action.shape),"fps":fps,"frames":len(action),"indices":{"left_arm":[0,1,2,3,4,5],"right_arm":[7,8,9,10,11,12],"left_gripper":a.left_gripper_index,"right_gripper":a.right_gripper_index},"open_close_evidence":"tools/map_aloha_gripper_to_dex3.py: increasing=OPEN; ALOHA model aperture limits [0,0.044] m","sides":{s:{k:v for k,v in d.items() if k!="derivative"} for s,d in stats.items()},"event_order":sorted([dict(e,side=s) for s in stats for e in stats[s]["events"]],key=lambda x:x["frame"])}
 print(json.dumps(summary,indent=2))
 if a.inspect_only:return 0
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with (a.output_dir/"gripper_signals.csv").open("w",newline="") as f:
  cols=["frame","time_sec","left_gripper_raw","right_gripper_raw","left_gripper_filtered","right_gripper_filtered","left_derivative","right_derivative"]; w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for i in range(len(action)):w.writerow(dict(zip(cols,[i,t[i],raw['left'][i],raw['right'][i],filt['left'][i],filt['right'][i],stats['left']['derivative'][i],stats['right']['derivative'][i]])))
 atomic_json(a.output_dir/"gripper_events.json",{"event_order":summary["event_order"]}); atomic_json(a.output_dir/"analysis_summary.json",summary)
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,1,sharex=True,figsize=(12,7));
 for j,s in enumerate(("left","right")):
  ax[j].plot(t,raw[s],alpha=.35,label="raw");ax[j].plot(t,filt[s],label="filtered")
  for e in stats[s]["events"]:ax[j].axvline(t[e['frame']],color='k',alpha=.12)
  ax[j].set_ylabel(f"{s} aperture (m)");ax[j].legend()
 ax[-1].set_xlabel("time (s)");fig.tight_layout();fig.savefig(a.output_dir/"gripper_signals.png",dpi=160);plt.close(fig);return 0
if __name__=="__main__":raise SystemExit(main())
