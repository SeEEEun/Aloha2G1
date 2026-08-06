#!/usr/bin/env python3
"""Render ALOHA tool axes on model and the verified Episode-49 cam frame."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
from task_frame_registration import build
CFG=ROOT/'configs/aloha_tool_axes_calibration.sim.json';OUT=ROOT/'outputs/task_frame_registration';XML=Path('/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml');ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';IMG=ROOT/'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000'
def project(p,Rc,pc,w=640,h=480,fovy=65):
 q=Rc.T@(np.asarray(p)-pc);f=.5*h/np.tan(np.radians(fovy)/2);return np.array([w/2+f*q[0]/(-q[2]),h/2-f*q[1]/(-q[2])]) if q[2]<0 else np.array([np.nan,np.nan])
def main():
 p=argparse.ArgumentParser();p.add_argument('--side',choices=('left','right','both'),default='both');p.add_argument('--frame',type=int,default=0);p.add_argument('--model-view',action='store_true');p.add_argument('--video-view',action='store_true');p.add_argument('--save',action='store_true');p.add_argument('--approve',action='store_true');a=p.parse_args();build();c=json.load(open(CFG));sides=('left','right') if a.side=='both' else (a.side,)
 import mujoco,matplotlib.pyplot as plt
 import validate_smolvla_in_stationary_aloha_mujoco as av
 m,_=av.load_validated_model(XML);d=mujoco.MjData(m);raw=np.load(ACTION)['optimized_action'];q,_=av.mapped_qpos(raw);d.qpos[:]=q[a.frame];mujoco.mj_forward(m,d)
 phases=list(__import__('csv').DictReader(open(ROOT/'outputs/magsafe_gripper_phases.csv')))[a.frame]
 camid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'cam_high');pc=d.cam_xpos[camid].copy();Rc=d.cam_xmat[camid].reshape(3,3).copy();colors={'approach':('red',0),'closing':('lime',1),'lateral':('cyan',2)}
 rendered=[]
 for side in sides:
  bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'follower_{side}_link_6');R=d.xmat[bid].reshape(3,3);origin=d.xpos[bid]+R@np.array([.1487,0,-.00105]);base=project(origin,Rc,pc)
  with mujoco.Renderer(m,480,640) as rr:rr.update_scene(d,camera='cam_high');model=rr.render()
  rawimg=plt.imread(IMG/f'frame_{a.frame:06d}.png')
  for kind,img,suffix in (("model",model,'model_view'),('episode',rawimg,'video_view')):
   fig,ax=plt.subplots(figsize=(10,7.5));ax.imshow(img);ax.axis('off')
   for name,(col,i) in colors.items():end=project(origin+R[:,i]*.10,Rc,pc);ax.annotate('',end,base,arrowprops={'arrowstyle':'->','lw':3,'color':col});ax.text(*end,f'{side} {name} +{"XYZ"[i]}',color=col,weight='bold')
   # Jaw pad centers and tips are explicit XML evidence, shown around TCP.
   for y,label in ((.02083,'jaw +Y tip'),(-.02083,'jaw -Y tip')):
    pt=project(d.xpos[bid]+R@np.array([.09115,y,-.00105]),Rc,pc);ax.scatter(*pt,c='yellow',s=35);ax.text(*pt,label,color='yellow',fontsize=8)
   ax.text(.01,.99,f'frame {a.frame} | {side}\nraw={phases[f"{side}_gripper_raw"]} automatic={phases[f"{side}_phase"]}\nopening plane: local X-Z',transform=ax.transAxes,va='top',color='white',bbox={'facecolor':'black','alpha':.75})
   path=OUT/'approval_views'/f'aloha_{side}_tool_axes_{suffix}_frame_{a.frame:06d}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig);rendered.append(str(path))
 if a.save:(OUT/'aloha_tool_axes_candidate.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'frame':a.frame,'rendered':rendered,'axes':{s:c[s] for s in sides}},indent=2)+'\n')
 if a.approve:
  pth=OUT/'aloha_tool_axes_candidate.json'
  if not pth.exists():raise SystemExit('Save candidate before approval.')
  z=json.load(open(pth));z['status']='ALOHA_TOOL_AXES_APPROVED';z['approved_at']=datetime.now(timezone.utc).isoformat();(OUT/'aloha_tool_axes.approved.json').write_text(json.dumps(z,indent=2)+'\n')
 print(json.dumps({'frame':a.frame,'rendered':rendered,'raw_and_phase':{k:phases[k] for k in phases if 'gripper' in k or 'phase' in k}},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
