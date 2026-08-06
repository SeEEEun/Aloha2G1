#!/usr/bin/env python3
"""Interactive active-G1 arm-only kinematic viewer; no physics stepping or hardware."""
import argparse,json,sys,time
from pathlib import Path
import mujoco,mujoco.viewer,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import retarget_episode49_optimized_action_to_g1 as core
import validate_g1_targets_and_sparse_ik as ik
from render_restored_arm_robot_comparison import add_scene,VIEWS
ROOT=Path('/home/jbnu/aloha_g1_dataset')
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--speed',type=float,default=.25);p.add_argument('--camera',choices=VIEWS,default='overview');a=p.parse_args()
z=np.load(a.input.resolve(),allow_pickle=False);q=z['g1_arm_joint_trajectory'];fps=float(z['fps']);root=np.asarray(z['g1_root_position']);reg=json.load(open(ROOT/'configs/magsafe_task_frame_registration.sim.json'));R=np.asarray(reg['T_scene_from_g1_base'])[:3,:3];quat=ik.mat_to_quat_wxyz(R)
if bool(z['real_robot_command_allowed']):raise RuntimeError('real robot trajectory refused')
info=ik.validate_model(core.G1_XML);m=info['model'];d=mujoco.MjData(m)
with mujoco.viewer.launch_passive(m,d) as v:
 spec=VIEWS[a.camera];v.cam.azimuth,v.cam.elevation,v.cam.distance=spec[:3];v.cam.lookat[:]=spec[3];add_scene(type('R',(),{'scene':v.user_scn})())
 for row in q:
  if not v.is_running():break
  ik.assign_arm_qpos(d,info['stand_qpos'],info['arm_qpos_ids'],row);d.qpos[:3]=root;d.qpos[3:7]=quat;d.qvel[:]=0;mujoco.mj_forward(m,d);v.sync();time.sleep(1/(fps*a.speed))
