#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');BASE=ROOT/'outputs/aloha_source_validation/episode49/hardware_preflight'
def main():
 p=argparse.ArgumentParser();p.add_argument('--trial-dir',type=Path);p.add_argument('--latest',action='store_true');a=p.parse_args()
 trial=a.trial_dir
 if a.latest:
  found=sorted((x for x in BASE.glob('*') if x.is_dir()),key=lambda x:x.stat().st_mtime);trial=found[-1] if found else None
 if trial is None:raise FileNotFoundError('no hardware-preflight result')
 with np.load(trial/'actual_state.npz',allow_pickle=False) as z:s=z['observation_state'];tm=z['timestamp_monotonic'];names=z['joint_names'].astype(str)
 delta=json.loads((trial/'current_to_action0.json').read_text());transition=json.loads((trial/'planned_start_transition_metrics.json').read_text());dt=np.diff(tm)
 checks={'shape_14':s.ndim==2 and s.shape[1]==14,'finite':bool(np.isfinite(s).all()),'joint_order':names.tolist(),'samples':len(s),'state_period_mean_s':float(dt.mean()) if len(dt) else None,'state_period_max_s':float(dt.max()) if len(dt) else None,'gripper_actual_ranges_m':{'left':[float(s[:,6].min()),float(s[:,6].max())],'right':[float(s[:,13].min()),float(s[:,13].max())]},'current_to_action0_l2':delta['l2_norm'],'transition_executed':transition['executed']}
 status='REQUIRES_HUMAN_REVIEW' if checks['shape_14'] and checks['finite'] and transition['executed'] is False else 'BLOCKED'
 (trial/'preflight_review.md').write_text('# ALOHA hardware preflight review\n\nStatus: **'+status+'**\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n\nNo replay or start-transition command was sent. A0 PASS requires human review and evidence.\n');print(trial/'preflight_review.md');return 0 if status!='BLOCKED' else 2
if __name__=='__main__':raise SystemExit(main())
