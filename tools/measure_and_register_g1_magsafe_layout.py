#!/usr/bin/env python3
"""Create, validate, and render a human-measured G1 MagSafe layout."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');DEFAULT=ROOT/'configs/g1_magsafe_layout.measured.json';FRAMES=ROOT/'configs/g1_magsafe_task_frames.measured.json';OUT=ROOT/'outputs/g1_task_registration'
FIELDS=('g1_base_pose','table_origin_pose','table_measurement_reference_pose','phone_center_pose','phone_landscape_orientation','accessory_center_pose','charger_center_pose','charger_surface_normal','accessory_placement_target_pose','table_height_m')
def template():
 d={'status':'MEASURED_NOT_APPROVED','root_forward_offset_m':.20,'units':'meters and quaternion_wxyz','measurement_photo_files':[],'operator':None,'measured_at':None}
 for k in FIELDS:d[k]=None
 d['camera_pose']=None;return d
def validate(d):
 errors=[]
 if d.get('root_forward_offset_m')!=.20:errors.append('G1 root forward offset must remain +0.20 m')
 for k in FIELDS:
  if d.get(k) is None:errors.append(f'missing human measurement: {k}')
 if not d.get('measurement_photo_files'):errors.append('measurement photos required')
 if d.get('status') not in ('MEASURED_NOT_APPROVED','APPROVED_BY_HUMAN'):errors.append('invalid status')
 return errors
def pose_matrix(p):
 xyz=np.asarray(p['position_xyz_m'],float);w,x,y,z=np.asarray(p['quaternion_wxyz'],float);n=np.linalg.norm([w,x,y,z]);w,x,y,z=np.asarray([w,x,y,z])/n;R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=xyz;return T
def apply(d):
 errors=validate(d)
 if errors:raise ValueError('; '.join(errors))
 mapping={'g1_base':'g1_base_pose','table':'table_origin_pose','phone':'phone_center_pose','accessory':'accessory_center_pose','charger':'charger_center_pose','accessory_placement':'accessory_placement_target_pose'}
 frames={k:pose_matrix(d[v]).tolist() for k,v in mapping.items()};frames.update({'left_hand_tool':'DYNAMIC_FROM_G1_ACTUAL_FK','right_hand_tool':'DYNAMIC_FROM_G1_ACTUAL_FK','left_palm':'DYNAMIC_FROM_DEX3_FK','right_palm':'DYNAMIC_FROM_DEX3_FK'})
 result={'status':'TASK_FRAMES_MEASURED_NOT_APPROVED','root_forward_offset_m':.20,'layout_source':str(DEFAULT),'frames_world':frames,'semantic_axes':{'phone_longitudinal':d['phone_landscape_orientation'],'phone_surface_normal':d['phone_center_pose'].get('surface_normal'),'accessory_removal_direction':d['accessory_center_pose'].get('removal_direction'),'charger_surface_normal':d['charger_surface_normal'],'aloha_approach_and_closing':'REUSE configs/aloha_tool_axes_calibration.sim.json WITH HUMAN REVIEW','dex3_palm_and_thumb_index':'REQUIRES_REAL_PRIMITIVE_AND_MODEL_REVIEW'}}
 FRAMES.write_text(json.dumps(result,indent=2)+'\n');OUT.mkdir(parents=True,exist_ok=True)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 pts=np.array([np.asarray(d[mapping[k]]['position_xyz_m'],float) for k in ('g1_base','table','phone','accessory','charger','accessory_placement')]);labels=list(('g1_base','table','phone','accessory','charger','accessory_placement'))
 for name,axes in {'front':(0,2),'side':(1,2),'top':(0,1),'semantic_frames':(0,1)}.items():
  fig,ax=plt.subplots();ax.scatter(pts[:,axes[0]],pts[:,axes[1]]);[ax.text(p[axes[0]],p[axes[1]],l) for p,l in zip(pts,labels)];ax.axis('equal');ax.grid();fig.savefig(OUT/f'{name}.png',dpi=140);plt.close(fig)
 (OUT/'task_frame_report.md').write_text('# Measured G1 task frames\n\nStatus: **TASK_FRAMES_MEASURED_NOT_APPROVED**\n\nNumeric transforms and four rendered views require human approval.\n');return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('template','validate','apply'),required=True);p.add_argument('--input',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args()
 if a.mode=='template':
  if a.output.exists():raise FileExistsError(a.output)
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(template(),indent=2)+'\n');print(a.output);return 0
 d=json.loads(a.input.read_text());errors=validate(d)
 if a.mode=='validate':print(json.dumps({'status':'BLOCKED' if errors else 'VALID_MEASURED_NOT_APPROVED','errors':errors},indent=2));return 2 if errors else 0
 print(json.dumps(apply(d),indent=2));return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except (ValueError,FileNotFoundError,FileExistsError) as e:print(f'BLOCKED: {e}',file=sys.stderr);raise SystemExit(2)
