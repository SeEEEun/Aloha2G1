#!/usr/bin/env python3
"""Compose an existing G1 arm trajectory with simulation-only Dex3 phases."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
import numpy as np
ROOT=Path("/home/jbnu/aloha_g1_dataset")
ARM=ROOT/"converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz"
PHASES=ROOT/"outputs/magsafe_gripper_phases.csv"; PRIMS=ROOT/"configs/dex3_magsafe_grasp_primitives.sim.json"; XML=Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml"); OUT=ROOT/"outputs/g1_magsafe_arm_dex3_full_trajectory.npz"
MAP={"left":{p:f"LEFT_PHONE_{p}" for p in ("OPEN","PREGRASP","GRASP","HOLD","RELEASE")},"right":{p:f"RIGHT_ACCESSORY_{p}" for p in ("OPEN","PREGRASP","GRASP","HOLD","RELEASE")}}
def cli():
 p=argparse.ArgumentParser();p.add_argument('--arm-trajectory',type=Path,default=ARM);p.add_argument('--phases',type=Path,default=PHASES);p.add_argument('--primitives',type=Path,default=PRIMS);p.add_argument('--xml',type=Path,default=XML);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--fps',type=float);p.add_argument('--transition-frames',type=int,default=6);p.add_argument('--transition-seconds',type=float);p.add_argument('--interpolation',choices=('linear','minjerk'),default='minjerk');p.add_argument('--no-interpolation',action='store_true');return p.parse_args()
def blend(x):return x if MODE=='linear' else 10*x**3-15*x**4+6*x**5
MODE='minjerk'
def interpolate(q,phase,n,mode):
 global MODE;MODE=mode;out=q.copy();reports=[]
 for start in np.flatnonzero(phase[1:]!=phase[:-1])+1:
  available=int(np.flatnonzero(np.r_[phase[start:]!=phase[start],True])[0]);use=min(n,available)
  if use<n:print(f"WARNING: transition at {start} shortened {n}->{use} frames")
  origin=out[start-1].copy();endpoint=q[start].copy()
  for k in range(use):out[start+k]=origin+blend((k+1)/use)*(endpoint-origin)
  reports.append({'start_frame':int(start),'end_frame':int(start+use-1),'requested_frames':n,'actual_frames':use,'from':str(phase[start-1]),'to':str(phase[start])})
 return out,reports
def main():
 a=cli();cfg=json.loads(a.primitives.read_text())
 if cfg.get('source')!='simulation_placeholder' or cfg.get('authoritative_for_real_robot') is not False or cfg.get('real_robot_command_allowed') is not False:raise RuntimeError('primitive config must be explicitly simulation-only')
 with np.load(a.arm_trajectory,allow_pickle=False) as z:
  if 'g1_arm_joint_trajectory' not in z.files:raise ValueError('missing g1_arm_joint_trajectory')
  arm=z['g1_arm_joint_trajectory'].astype(float);arm_names=z['arm_joint_names'].astype(str);fps=float(a.fps or z['fps']);source_ts=z['source_timestamp'].astype(float) if 'source_timestamp' in z.files else np.arange(len(arm))/fps
 with a.phases.open() as f:rows=list(csv.DictReader(f))
 if len(rows)!=len(arm):raise ValueError(f'mismatched frame count: phases={len(rows)} arm={len(arm)}')
 left_phase=np.array([r['left_phase'] for r in rows]);right_phase=np.array([r['right_phase'] for r in rows]);left_prim=np.array([MAP['left'][p] for p in left_phase]);right_prim=np.array([MAP['right'][p] for p in right_phase])
 def qseq(names,side):
  out=[];expected=cfg['joint_names'][f'{side}_dex3']
  for name in names:
   item=cfg['primitives'].get(str(name));
   if not item:raise ValueError(f'missing primitive {name}')
   if item['authoritative_side']!=side or item['joint_names']!=expected:raise ValueError(f'{name}: wrong side/order')
   out.append(item['qpos'])
  return np.asarray(out,float),np.asarray(expected)
 left,left_names=qseq(left_prim,'left');right,right_names=qseq(right_prim,'right')
 n=0 if a.no_interpolation else int(round(a.transition_seconds*fps)) if a.transition_seconds is not None else a.transition_frames
 if n<0:raise ValueError('transition length must be nonnegative')
 lrep=rrep=[]
 if n:left,lrep=interpolate(left,left_phase,n,a.interpolation);right,rrep=interpolate(right,right_phase,n,a.interpolation)
 import mujoco
 model=mujoco.MjModel.from_xml_path(str(a.xml));base=model.key_qpos[0].copy();full=np.repeat(base[None],len(arm),axis=0)
 def add(names,values):
  ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(x)) for x in names]);
  if np.any(ids<0):raise ValueError(f'missing XML joints: {np.asarray(names)[ids<0].tolist()}')
  full[:,model.jnt_qposadr[ids]]=values;return model.jnt_qposadr[ids]
 arm_addr=add(arm_names,arm);left_addr=add(left_names,left);right_addr=add(right_names,right)
 if not np.array_equal(full[:,arm_addr],arm):raise AssertionError('arm qpos preservation failed')
 timestamps=np.asarray([float(r['time_sec']) for r in rows]);source_frame=np.asarray([int(r['frame']) for r in rows]);gl=np.asarray([float(r['left_gripper_raw']) for r in rows]);gr=np.asarray([float(r['right_gripper_raw']) for r in rows]);joint_names=np.array([mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '' for j in range(model.njnt) for _ in range(1) if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE])
 full_names=np.array(['']*model.nq,dtype='U64')
 for j in range(model.njnt):
  if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
   for k,s in enumerate(('x','y','z','qw','qx','qy','qz')):full_names[model.jnt_qposadr[j]+k]=f'{mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)}:{s}'
  else:full_names[model.jnt_qposadr[j]]=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
 payload=dict(timestamps=timestamps,fps=np.array(fps),full_qpos=full,arm_qpos=arm,left_dex3_qpos=left,right_dex3_qpos=right,full_joint_names=full_names,arm_joint_names=arm_names,left_dex3_joint_names=left_names,right_dex3_joint_names=right_names,left_phase=left_phase,right_phase=right_phase,left_primitive=left_prim,right_primitive=right_prim,source_aloha_frame=source_frame,source_gripper_left=gl,source_gripper_right=gr,primitive_config_path=np.array(str(a.primitives.resolve())),primitive_source=np.array(cfg['source']),authoritative_for_real_robot=np.array(False),real_robot_command_allowed=np.array(False),source_arm_trajectory_path=np.array(str(a.arm_trajectory.resolve())),source_arm_trajectory_key=np.array('g1_arm_joint_trajectory'),source_arm_timestamps=source_ts,xml_path=np.array(str(a.xml.resolve())))
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.npz.tmp');
 with tmp.open('wb') as f:np.savez_compressed(f,**payload)
 os.replace(tmp,a.output)
 step=np.abs(np.diff(np.c_[left,right],axis=0));vel=step*fps;acc=np.abs(np.diff(np.c_[left,right],n=2,axis=0))*fps**2
 report={'output':str(a.output.resolve()),'frames':len(arm),'fps':fps,'duration_s':float(timestamps[-1]-timestamps[0]),'shapes':{k:list(payload[k].shape) for k in ('full_qpos','arm_qpos','left_dex3_qpos','right_dex3_qpos')},'arm_preserved_exactly':True,'interpolation':{'enabled':bool(n),'method':a.interpolation,'frames':n,'left_transitions':lrep,'right_transitions':rrep},'finger_metrics':{'max_step_rad':float(step.max(initial=0)),'max_velocity_rad_s':float(vel.max(initial=0)),'max_acceleration_rad_s2':float(acc.max(initial=0))},'simulation_only':True,'authoritative_for_real_robot':False}
 a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
