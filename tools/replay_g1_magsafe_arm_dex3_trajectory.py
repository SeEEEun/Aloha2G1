#!/usr/bin/env python3
"""Kinematic-only full G1 arm+Dex3 replay; physics is intentionally gated."""
from __future__ import annotations
import argparse,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');DEFAULT=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
def cli():
 p=argparse.ArgumentParser();p.add_argument('--trajectory',type=Path,default=DEFAULT);p.add_argument('--xml',type=Path,default=XML);p.add_argument('--mode',choices=('kinematic','physics'),default='kinematic');p.add_argument('--loop',action='store_true');p.add_argument('--speed',type=float,default=1);p.add_argument('--start-frame',type=int,default=0);p.add_argument('--end-frame',type=int);p.add_argument('--show-phase',action='store_true');p.add_argument('--save-video',type=Path);p.add_argument('--headless',action='store_true');p.add_argument('--validate-first',action='store_true');return p.parse_args()
def main():
 a=cli();print('SIMULATION PLACEHOLDER DEX3 PRIMITIVES\nNOT VALIDATED FOR REAL ROBOT')
 if a.mode=='physics':print('PHYSICS MODE NOT_IMPLEMENTED: use the existing Isaac Lab controller replay');return 3
 if a.validate_first:
  r=subprocess.run(['python3',str(ROOT/'tools/validate_g1_arm_dex3_trajectory.py'),'--trajectory',str(a.trajectory),'--xml',str(a.xml)]); 
  if r.returncode:raise RuntimeError('validation failed')
 import mujoco
 with np.load(a.trajectory,allow_pickle=False) as z:q=z['full_qpos'];fps=float(z['fps']);lp=z['left_phase'].astype(str);rp=z['right_phase'].astype(str);lprim=z['left_primitive'].astype(str);rprim=z['right_primitive'].astype(str)
 if bool(np.load(a.trajectory,allow_pickle=False)['authoritative_for_real_robot']):raise RuntimeError('expected simulation-only trajectory')
 end=len(q)-1 if a.end_frame is None else a.end_frame
 if not 0<=a.start_frame<=end<len(q):raise ValueError('invalid frame range')
 model=mujoco.MjModel.from_xml_path(str(a.xml));data=mujoco.MjData(model)
 def setq(i):data.qpos[:]=q[i];data.qvel[:]=0;mujoco.mj_forward(model,data)
 transitions=set((np.flatnonzero((lp[1:]!=lp[:-1])|(rp[1:]!=rp[:-1]))+1).tolist())
 if a.save_video:
  import cv2;mujoco.mj_forward(model,data);width,height=640,480;renderer=mujoco.Renderer(model,height,width)
  a.save_video.parent.mkdir(parents=True,exist_ok=True);w=cv2.VideoWriter(str(a.save_video),cv2.VideoWriter_fourcc(*'mp4v'),fps*a.speed,(width,height))
  for i in range(a.start_frame,end+1):
   setq(i);renderer.update_scene(data,camera='track' if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'track')>=0 else -1);im=renderer.render().copy();cv2.putText(im,f'{i} L:{lp[i]} R:{rp[i]}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,40,40),2);w.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
   if i in transitions:print(f'TRANSITION frame={i} left={lp[i]} right={rp[i]}')
  w.release();renderer.close();print(f'Saved {a.save_video}')
  if a.headless:return 0
 if a.headless:
  for i in range(a.start_frame,end+1):setq(i)
  print(f'headless mj_forward success: {end-a.start_frame+1} frames');return 0
 import mujoco.viewer
 while True:
  with mujoco.viewer.launch_passive(model,data) as viewer:
   for i in range(a.start_frame,end+1):
    if not viewer.is_running():return 0
    tick=time.monotonic();setq(i);viewer.sync()
    if a.show_phase and (i%30==0 or i in transitions):print(f'frame={i} left={lp[i]}:{lprim[i]} right={rp[i]}:{rprim[i]} | SIMULATION ONLY')
    time.sleep(max(0,1/(fps*a.speed)-(time.monotonic()-tick)))
  if not a.loop:break
 return 0
if __name__=='__main__':raise SystemExit(main())
