#!/usr/bin/env python3
"""Offline fixed-transform calibration viewer; never changes object poses."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
from task_frame_registration import build
from render_registration_approval_views import render_all
CFG=ROOT/'configs/magsafe_task_frame_registration.sim.json';OUT=ROOT/'outputs/task_frame_registration'
def main():
 p=argparse.ArgumentParser();p.add_argument('--view',choices=('front','side','top'),default='front');p.add_argument('--translation-mm',nargs=3,type=float,default=(0,0,0));p.add_argument('--yaw-deg',type=float,default=0);p.add_argument('--save-candidate',action='store_true');p.add_argument('--approve',action='store_true');a=p.parse_args();build();c=json.load(open(CFG));base=np.array(c['T_scene_from_g1_base'],float);yaw=np.radians(a.yaw_deg);R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
 x=base.copy();x[:3,:3]=R@x[:3,:3];x[:3,3]+=np.asarray(a.translation_mm)/1000
 hist={'created_at':datetime.now(timezone.utc).isoformat(),'base_config':str(CFG),'fixed_transform_candidate':x.tolist(),'translation_delta_mm':list(a.translation_mm),'yaw_delta_deg':a.yaw_deg,'objects_modified':False,'reason':'USER_MUST_SUPPLY_REASON_DURING_FORMAL_APPROVAL'}
 if a.save_candidate:(OUT/'g1_scene_registration_candidate.json').write_text(json.dumps(hist,indent=2)+'\n')
 if a.approve:
  if not (OUT/'g1_scene_registration_candidate.json').exists():raise SystemExit('Save a candidate before approval.')
  hist=json.load(open(OUT/'g1_scene_registration_candidate.json'));hist['status']='G1_SCENE_REGISTRATION_APPROVED';hist['approved_at']=datetime.now(timezone.utc).isoformat();(OUT/'g1_scene_registration.approved.json').write_text(json.dumps(hist,indent=2)+'\n')
 # Approval evidence always renders all fixed views with the actual G1 visual
 # meshes and immutable scene geometry. CLI view only selects which path to print.
 met=render_all(OUT/'approval_views');hist['rendered_metrics']=met;hist['requested_view_path']=str(OUT/'approval_views'/f'g1_scene_{a.view}.png');print(json.dumps(hist,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
