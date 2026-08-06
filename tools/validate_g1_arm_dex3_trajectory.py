#!/usr/bin/env python3
"""Validate composed G1 arm + simulation Dex3 trajectory with MuJoCo."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');DEFAULT=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';OUT=ROOT/'outputs/g1_magsafe_arm_dex3_validation.json'
def cli():
 p=argparse.ArgumentParser();p.add_argument('--trajectory',type=Path,default=DEFAULT);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--xml',type=Path,default=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml'));return p.parse_args()
def main():
 a=cli();import mujoco
 with np.load(a.trajectory,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 required=('timestamps','fps','full_qpos','arm_qpos','left_dex3_qpos','right_dex3_qpos','full_joint_names','arm_joint_names','left_dex3_joint_names','right_dex3_joint_names','left_phase','right_phase','left_primitive','right_primitive','source_aloha_frame','primitive_source','authoritative_for_real_robot','real_robot_command_allowed','source_arm_trajectory_path','source_arm_trajectory_key')
 errors=[f'missing {k}' for k in required if k not in d];model=mujoco.MjModel.from_xml_path(str(a.xml));n=len(d['full_qpos']);fps=float(d['fps'])
 shapes={'full_qpos':list(d['full_qpos'].shape),'arm_qpos':list(d['arm_qpos'].shape),'left_dex3_qpos':list(d['left_dex3_qpos'].shape),'right_dex3_qpos':list(d['right_dex3_qpos'].shape)}
 if d['full_qpos'].shape!=(n,model.nq):errors.append('full_qpos shape mismatch')
 if d['arm_qpos'].shape!=(n,14) or d['left_dex3_qpos'].shape!=(n,7) or d['right_dex3_qpos'].shape!=(n,7):errors.append('component shape mismatch')
 if any(not np.isfinite(d[k]).all() for k in ('full_qpos','arm_qpos','left_dex3_qpos','right_dex3_qpos','timestamps')):errors.append('NaN/Inf')
 if len(d['full_joint_names'])!=model.nq:errors.append('full joint name count mismatch')
 if len(d['timestamps'])!=n or not np.all(np.diff(d['timestamps'])>0):errors.append('timestamps not monotonic')
 dt=np.diff(d['timestamps']);fps_error=float(np.max(np.abs(dt-1/fps),initial=0));
 if fps_error>2e-5:errors.append('fps/timestamp inconsistency')
 def addr(names):
  ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(x)) for x in names]);
  if np.any(ids<0):raise ValueError('joint name/order includes XML-missing name')
  return model.jnt_qposadr[ids]
 aa,la,ra=addr(d['arm_joint_names']),addr(d['left_dex3_joint_names']),addr(d['right_dex3_joint_names'])
 if not np.array_equal(d['full_qpos'][:,aa],d['arm_qpos']):errors.append('composed arm differs from stored arm_qpos')
 with np.load(str(d['source_arm_trajectory_path']),allow_pickle=False) as z:source=z[str(d['source_arm_trajectory_key'])]
 arm_preserved=source.shape==d['arm_qpos'].shape and np.array_equal(source,d['arm_qpos']);
 if not arm_preserved:errors.append('arm trajectory source not preserved exactly')
 if not np.array_equal(d['full_qpos'][:,la],d['left_dex3_qpos']) or not np.array_equal(d['full_qpos'][:,ra],d['right_dex3_qpos']):errors.append('Dex3 placement mismatch')
 valid={'OPEN','PREGRASP','GRASP','HOLD','RELEASE'}
 if any(str(x) not in valid for x in np.r_[d['left_phase'],d['right_phase']]):errors.append('missing/unknown phase')
 if any(not str(x) for x in np.r_[d['left_primitive'],d['right_primitive']]):errors.append('missing primitive name')
 sim=(str(d['primitive_source'])=='simulation_placeholder' and not bool(d['authoritative_for_real_robot']) and not bool(d['real_robot_command_allowed']));
 if not sim:errors.append('simulation-only metadata/gate absent')
 limit_violations=0
 for j in range(model.njnt):
  if model.jnt_limited[j] and model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:
   q=d['full_qpos'][:,model.jnt_qposadr[j]];limit_violations+=int(np.count_nonzero((q<model.jnt_range[j,0]-1e-9)|(q>model.jnt_range[j,1]+1e-9)))
 if limit_violations:errors.append(f'joint limit violations={limit_violations}')
 data=mujoco.MjData(model);mj_fail=0;cats=Counter();frame_contacts=[]
 def group(body):
  if body.startswith('left_hand'):return 'left_hand'
  if body.startswith('right_hand'):return 'right_hand'
  if body=='torso_link':return 'torso'
  if any(x in body for x in ('shoulder','elbow','wrist')):return 'arm'
  return 'other'
 for i,q in enumerate(d['full_qpos']):
  try:data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(model,data)
  except Exception:mj_fail+=1;continue
  pairs=[]
  for c in data.contact:
   b=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or 'world' for g in (c.geom1,c.geom2)];g=sorted((group(b[0]),group(b[1])));key='-'.join(g);cats[key]+=1;pairs.append('|'.join(sorted(b)))
  frame_contacts.append({'frame':i,'contact_count':len(pairs),'pairs':';'.join(sorted(set(pairs)))})
 if mj_fail:errors.append(f'mj_forward failures={mj_fail}')
 fingers=np.c_[d['left_dex3_qpos'],d['right_dex3_qpos']];step=np.abs(np.diff(fingers,axis=0));vel=step*fps;acc=np.abs(np.diff(fingers,n=2,axis=0))*fps**2
 trans=np.flatnonzero((d['left_phase'][1:]!=d['left_phase'][:-1])|(d['right_phase'][1:]!=d['right_phase'][:-1]))+1;trans_jump=float(np.max(np.abs(fingers[trans]-fingers[trans-1]),initial=0))
 open_grasp={}
 for side in ('left','right'):
  phases=d[f'{side}_phase'];q=d[f'{side}_dex3_qpos'];o=q[phases=='OPEN'];g=q[phases=='GRASP'];open_grasp[side]=float(np.max(np.abs(np.median(o,0)-np.median(g,0)))) if len(o) and len(g) else None
 report={'status':'PASS' if not errors else 'FAIL','errors':errors,'trajectory':str(a.trajectory.resolve()),'frames':n,'duration_s':float(d['timestamps'][-1]-d['timestamps'][0]),'fps':fps,'fps_max_error_s':fps_error,'shapes':shapes,'nan_inf_count':sum(int((~np.isfinite(d[k])).sum()) for k in ('full_qpos','timestamps')),'joint_limit_violation_count':limit_violations,'arm_preserved_exactly':arm_preserved,'source_frame_correspondence':bool(np.array_equal(d['source_aloha_frame'],np.arange(n))),'simulation_config':sim,'real_robot_use_allowed':False,'mj_forward_failures':mj_fail,'contact_check':'AVAILABLE','contact_category_counts':dict(cats),'frames_with_contacts':sum(x['contact_count']>0 for x in frame_contacts),'finger_metrics':{'max_step_rad':float(step.max(initial=0)),'max_velocity_rad_s':float(vel.max(initial=0)),'max_acceleration_rad_s2':float(acc.max(initial=0)),'phase_transition_max_step_rad':trans_jump,'limits':'MEASURED_ONLY_NO_VERIFIED_DEX3_DYNAMIC_LIMITS'},'open_grasp_max_difference_rad':open_grasp,'grasp_hold_note':'semantic phases may legally share identical qpos'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 with a.output.with_suffix('.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=('frame','contact_count','pairs'));w.writeheader();w.writerows(frame_contacts)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,1,sharex=True,figsize=(12,6));ax[0].plot(d['left_dex3_qpos']);ax[0].set_title('left Dex3');ax[1].plot(d['right_dex3_qpos']);ax[1].set_title('right Dex3');fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=150);plt.close(fig)
 print(json.dumps(report,indent=2));return 0 if not errors else 2
if __name__=='__main__':raise SystemExit(main())
