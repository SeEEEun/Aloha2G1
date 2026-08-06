#!/usr/bin/env python3
"""Object-free Isaac Lab replay of an offline simulation-only G1+Dex3 NPZ.

Actuator settings are reused verbatim from replay_smolvla_episode49_g1.py.
No hardware, DDS, or robot command modules are imported.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

ROOT=Path('/home/jbnu/aloha_g1_dataset');DEFAULT=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';OUT=ROOT/'outputs';G1_USD=Path('/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd')
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=DEFAULT);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--mode',choices=('arm-only','arm-dex3'),default='arm-dex3');p.add_argument('--max-frames',type=int);p.add_argument('--settle-seconds',type=float,default=1);p.add_argument('--speed',type=float,default=1);p.add_argument('--dry-run',action='store_true');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
def save_json(path,data):path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
def load(path):
 with np.load(path,allow_pickle=False) as z:
  req=('arm_qpos','left_dex3_qpos','right_dex3_qpos','arm_joint_names','left_dex3_joint_names','right_dex3_joint_names','fps','primitive_source','authoritative_for_real_robot','real_robot_command_allowed');miss=[k for k in req if k not in z.files]
  if miss:raise RuntimeError(f'missing trajectory keys {miss}')
  d={k:z[k] for k in req}
 if str(d['primitive_source'])!='simulation_placeholder' or bool(d['authoritative_for_real_robot']) or bool(d['real_robot_command_allowed']):raise RuntimeError('SIMULATION-ONLY SAFETY GATE FAILED')
 if [d[k].shape[1] for k in ('arm_qpos','left_dex3_qpos','right_dex3_qpos')]!=[14,7,7]:raise RuntimeError('trajectory shapes invalid')
 return d
trajectory=load(args.input.resolve())
if args.mode=='arm-only':
 trajectory['left_dex3_qpos']=np.repeat(trajectory['left_dex3_qpos'][:1],len(trajectory['arm_qpos']),axis=0)
 trajectory['right_dex3_qpos']=np.repeat(trajectory['right_dex3_qpos'][:1],len(trajectory['arm_qpos']),axis=0)
if args.dry_run:
 print(json.dumps({'status':'DRY_RUN_PASS','mode':args.mode,'input':str(args.input.resolve()),'frames':len(trajectory['arm_qpos']),'hardware_commands_sent':False,'dds_initialized':False},indent=2));raise SystemExit(0)
launcher=AppLauncher(args);simulation_app=launcher.app
def main():
 import torch,omni.usd
 from pxr import Gf,Usd,UsdGeom
 from isaaclab.assets import Articulation,ArticulationCfg
 from isaaclab.actuators import ImplicitActuatorCfg
 from isaaclab.sim import SimulationCfg,SimulationContext
 arm=trajectory['arm_qpos'].astype(np.float32);left=trajectory['left_dex3_qpos'].astype(np.float32);right=trajectory['right_dex3_qpos'].astype(np.float32)
 if args.max_frames:arm,left,right=arm[:args.max_frames],left[:args.max_frames],right[:args.max_frames]
 stage=args.output_dir/'isaaclab_g1_arm_dex3_object_free.usda';stage.parent.mkdir(parents=True,exist_ok=True)
 usd=Usd.Stage.CreateNew(str(stage));UsdGeom.SetStageMetersPerUnit(usd,1.0);UsdGeom.SetStageUpAxis(usd,UsdGeom.Tokens.z);world=UsdGeom.Xform.Define(usd,'/World').GetPrim();usd.SetDefaultPrim(world);robot_xform=UsdGeom.Xform.Define(usd,'/World/G1');asset=UsdGeom.Xform.Define(usd,'/World/G1/Asset').GetPrim();asset.GetReferences().AddReference(str(G1_USD));robot_xform.AddTranslateOp().Set(Gf.Vec3d(0,0,0));usd.GetRootLayer().Save()
 if not omni.usd.get_context().open_stage(str(stage)):raise RuntimeError(f'cannot open {stage}')
 sim=SimulationContext(SimulationCfg(device=args.device));robot=Articulation(ArticulationCfg(prim_path='/World/G1/Asset/root_joint',spawn=None,actuators={
  'arms':ImplicitActuatorCfg(joint_names_expr=[r'(left|right)_(shoulder|wrist)_.*_joint',r'(left|right)_elbow_joint'],effort_limit_sim=25.0,velocity_limit_sim=12.0,stiffness=100.0,damping=5.0),
  'dex3':ImplicitActuatorCfg(joint_names_expr=[r'(left|right)_hand_.*_joint'],effort_limit_sim=2.5,velocity_limit_sim=12.0,stiffness=50.0,damping=2.0)}));sim.reset()
 runtime=list(robot.data.joint_names);wanted=trajectory['arm_joint_names'].tolist()+trajectory['left_dex3_joint_names'].tolist()+trajectory['right_dex3_joint_names'].tolist();missing=[n for n in wanted if n not in runtime]
 if missing or len(set(wanted))!=28:raise RuntimeError(f'joint name remap refused: missing={missing}')
 ids=[runtime.index(n) for n in wanted];target=robot.data.default_joint_pos.torch.clone().to(robot.device,dtype=torch.float32);dt=sim.get_physics_dt();sub=max(1,int(round(1/(float(trajectory['fps'])*args.speed*dt))));settle=int(round(args.settle_seconds/dt));desired=np.c_[arm,left,right]
 target[0,ids]=torch.as_tensor(desired[0],device=robot.device);robot.write_joint_state_to_sim(target,torch.zeros_like(target));sim.reset()
 for _ in range(settle):robot.set_joint_position_target(target);robot.write_data_to_sim();sim.step(render=False);robot.update(dt)
 actual=[]
 for i,row in enumerate(desired):
  target[0,ids]=torch.as_tensor(row,device=robot.device)
  for _ in range(sub):robot.set_joint_position_target(target);robot.write_data_to_sim();sim.step(render=False);robot.update(dt)
  q=robot.data.joint_pos.torch[0,ids].detach().cpu().numpy()
  if not np.isfinite(q).all():raise RuntimeError(f'instability/nonfinite at frame {i}')
  actual.append(q)
 actual=np.asarray(actual);err=actual-desired;rmse=np.sqrt(np.mean(err*err,axis=0));args.output_dir.mkdir(parents=True,exist_ok=True)
 payload=dict(target_qpos=desired,actual_qpos=actual,joint_tracking_error=err,joint_names=np.asarray(wanted),per_joint_rmse=rmse,fps=trajectory['fps']);np.savez_compressed(args.output_dir/'isaaclab_g1_arm_dex3_replay_results.npz',**payload);np.savez_compressed(args.output_dir/'replay_results.npz',**payload)
 summary={'status':'PASS','mode':args.mode,'frames':len(desired),'duration_sec':len(desired)/float(trajectory['fps']),'object_scene':'NONE_FIXED_BASE_G1_ONLY','joint_mapping':'NAME_BASED','missing_joints':missing,'joint_names':wanted,'per_joint_rmse':dict(zip(wanted,rmse.tolist())),'overall_rmse':float(np.sqrt(np.mean(err*err))),'arm_rmse':float(np.sqrt(np.mean(err[:,:14]**2))),'max_error':float(np.max(np.abs(err))),'simulation_stability':'FINITE','contact_count':'NOT_AVAILABLE','left_dex3_rmse':float(np.sqrt(np.mean(err[:,14:21]**2))),'right_dex3_rmse':float(np.sqrt(np.mean(err[:,21:28]**2))),'hardware_commands_sent':False,'dds_initialized':False,'controller_source':str((Path(__file__).parent/'replay_smolvla_episode49_g1.py').resolve())}
 save_json(args.output_dir/'isaaclab_g1_arm_dex3_replay_summary.json',summary);save_json(args.output_dir/'replay_summary.json',summary)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,4));ax.plot(np.sqrt(np.mean(err[:,:14]**2,axis=1)));ax.set(xlabel='frame',ylabel='arm RMSE (rad)');fig.tight_layout();fig.savefig(args.output_dir/'tracking_error.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,4));ax.bar(np.arange(len(wanted)),rmse);ax.set(xlabel='name-mapped joint index',ylabel='RMSE (rad)');fig.tight_layout();fig.savefig(args.output_dir/'per_joint_rmse.png',dpi=150);plt.close(fig)
 print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 finally:simulation_app.close()
