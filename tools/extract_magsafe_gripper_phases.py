#!/usr/bin/env python3
"""Deterministic, stateful ALOHA gripper semantic phase extraction."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from magsafe_gripper_common import *
OUT=Path("/home/jbnu/aloha_g1_dataset/outputs/magsafe_gripper_phases.csv")

class GripperPhaseTracker:
 def __init__(self,open_threshold,close_threshold,min_dwell_frames=5,grasp_transition_frames=5,pregrasp_max_frames=30,increasing_is_open=True):
  self.open_threshold=float(open_threshold);self.close_threshold=float(close_threshold);self.min_dwell=int(min_dwell_frames);self.grasp_frames=int(grasp_transition_frames);self.pregrasp_max=int(pregrasp_max_frames);self.increasing_is_open=bool(increasing_is_open);self.reset(None)
 def reset(self,initial_value=None):
  self.phase=None;self.age=0;self.prev=None;self.candidate=None;self.candidate_age=0;self.frame=-1
  if initial_value is not None:self._initial(float(initial_value))
 def _open(self,v):return v>=self.open_threshold if self.increasing_is_open else v<=self.open_threshold
 def _closed(self,v):return v<=self.close_threshold if self.increasing_is_open else v>=self.close_threshold
 def _initial(self,v):self.phase="OPEN" if self._open(v) else "HOLD" if self._closed(v) else "PREGRASP";self.prev=v;self.age=0
 def update(self,value):
  v=float(value)
  if not np.isfinite(v):raise ValueError("non-finite gripper value")
  self.frame+=1
  if self.phase is None:self._initial(v);return self.get_state("episode_initial_value")
  delta=v-self.prev; close_motion=delta<0 if self.increasing_is_open else delta>0; open_motion=delta>0 if self.increasing_is_open else delta<0
  desired=None;reason=""
  if self.phase=="OPEN" and close_motion and not self._open(v):desired="PREGRASP";reason="close_direction_motion"
  elif self.phase=="PREGRASP" and self._closed(v):desired="GRASP";reason="close_threshold_crossed"
  elif self.phase=="PREGRASP" and self._open(v):desired="OPEN";reason="returned_to_open_region"
  elif self.phase=="GRASP" and self.age>=self.grasp_frames:desired="HOLD";reason="grasp_transition_elapsed"
  elif self.phase=="HOLD" and open_motion and not self._closed(v):desired="RELEASE";reason="open_direction_motion"
  elif self.phase=="RELEASE" and self._open(v):desired="OPEN";reason="open_threshold_crossed"
  elif self.phase=="RELEASE" and self._closed(v):desired="HOLD";reason="returned_to_close_region"
  # Debounce threshold changes; motion-derived PREGRASP/RELEASE is immediate but spikes revert naturally.
  threshold_change=desired in ("GRASP","OPEN")
  if desired and threshold_change:
   if self.candidate==desired:self.candidate_age+=1
   else:self.candidate,self.candidate_age=desired,1
   if self.candidate_age<self.min_dwell:desired=None
  else:self.candidate,self.candidate_age=None,0
  transition=""
  if desired:
   old=self.phase;self.phase=desired;self.age=0;transition=f"{old}->{desired}:{reason}"
  else:self.age+=1
  self.prev=v;return self.get_state(transition)
 def get_state(self,transition=""):return {"phase":self.phase,"age":self.age,"transition":transition,"frame":self.frame}
 def serialize_state(self):return {"phase":self.phase,"age":self.age,"prev":self.prev,"candidate":self.candidate,"candidate_age":self.candidate_age,"frame":self.frame}
 def load_state(self,state):
  for k in ("phase","age","prev","candidate","candidate_age","frame"):setattr(self,k,state[k])

def cli():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,default=DEFAULT_ACTION);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--analysis-config",type=Path,default=Path("/home/jbnu/aloha_g1_dataset/outputs/gripper_phase_analysis/analysis_summary.json"));p.add_argument("--fps",type=float);p.add_argument("--left-open-threshold",type=float);p.add_argument("--left-close-threshold",type=float);p.add_argument("--right-open-threshold",type=float);p.add_argument("--right-close-threshold",type=float);p.add_argument("--min-dwell-frames",type=int,default=5);p.add_argument("--grasp-transition-frames",type=int,default=5);p.add_argument("--pregrasp-max-frames",type=int,default=30);p.add_argument("--plot",action="store_true");p.add_argument("--inspect",action="store_true");return p.parse_args()
def main():
 a=cli();action,t,fps=load_action(a.input,LEFT_INDEX,RIGHT_INDEX,a.fps); config=json.loads(a.analysis_config.read_text()) if a.analysis_config.exists() else None
 def thresholds(side):
  o=getattr(a,f"{side}_open_threshold");c=getattr(a,f"{side}_close_threshold")
  if o is None or c is None:
   if not config:raise ValueError("thresholds missing and analysis config unavailable")
   o=o if o is not None else config["sides"][side]["open_threshold"];c=c if c is not None else config["sides"][side]["close_threshold"]
  if not o>c:raise ValueError(f"{side}: increasing-is-open requires open threshold > close threshold")
  return o,c
 filt={"left":smooth(action[:,LEFT_INDEX]),"right":smooth(action[:,RIGHT_INDEX])}; trackers={s:GripperPhaseTracker(*thresholds(s),a.min_dwell_frames,a.grasp_transition_frames,a.pregrasp_max_frames,True) for s in ('left','right')}; states={s:[] for s in trackers}
 for i in range(len(action)):
  for s in trackers:states[s].append(trackers[s].update(filt[s][i]))
 rows=[]
 for i in range(len(action)):rows.append({"frame":i,"time_sec":t[i],"left_gripper_raw":action[i,LEFT_INDEX],"right_gripper_raw":action[i,RIGHT_INDEX],"left_phase":states['left'][i]['phase'],"right_phase":states['right'][i]['phase'],"left_transition":states['left'][i]['transition'],"right_transition":states['right'][i]['transition'],"left_phase_age_frames":states['left'][i]['age'],"right_phase_age_frames":states['right'][i]['age']})
 valid={"OPEN","PREGRASP","GRASP","HOLD","RELEASE"};assert all(r['left_phase'] in valid and r['right_phase'] in valid for r in rows)
 events={s:[{"frame":i,"transition":x['transition']} for i,x in enumerate(states[s]) if x['transition']] for s in states};summary={"input":str(a.input.resolve()),"fps":fps,"frames":len(rows),"threshold_source":str(a.analysis_config.resolve()) if config else "CLI","thresholds":{s:{"open":trackers[s].open_threshold,"close":trackers[s].close_threshold,"direction":"increasing_is_open"} for s in trackers},"phase_runs":{s:[{"start":x,"end":y,"phase":p,"frames":y-x+1} for x,y,p in runs([v['phase'] for v in states[s]])] for s in states},"transition_counts":{s:len(events[s]) for s in states},"event_order":sorted([dict(e,side=s) for s in events for e in events[s]],key=lambda x:x['frame']),"validation":{"all_frames_labeled":True,"unknown_phase_count":0,"nan_inf_count":0,"multiple_transition_same_frame":False}}
 print(json.dumps(summary,indent=2));
 if a.inspect:return 0
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 atomic_json(a.output.with_name('magsafe_gripper_phase_events.json'),events);atomic_json(a.output.with_name('magsafe_gripper_phase_summary.json'),summary)
 if a.plot:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots(2,1,sharex=True,figsize=(12,7));codes={p:i for i,p in enumerate(valid)}
  for j,s in enumerate(('left','right')):ax[j].plot(t,action[:,LEFT_INDEX if s=='left' else RIGHT_INDEX],label='raw');ax[j].plot(t,filt[s],label='filtered');ax[j].set_title(s+' phases: '+', '.join(f"{e['frame']} {e['transition']}" for e in events[s]));ax[j].legend()
  fig.tight_layout();fig.savefig(a.output.with_name('magsafe_gripper_phases.png'),dpi=160);plt.close(fig)
 return 0
if __name__=='__main__':raise SystemExit(main())
