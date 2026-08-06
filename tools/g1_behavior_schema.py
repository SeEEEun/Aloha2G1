#!/usr/bin/env python3
"""Canonical g1_behavior_v1 schema and name-safe offline utilities."""
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
SCHEMA='g1_behavior_v1';ROLES={'generated_target','generated_executed','expert_actual','synthetic_test'};STATUS={'not_executed','executed','synthetic'}
REQUIRED_META={'schema_version','behavior_id','behavior_role','task_name','robot','hand','source_type','source_path','created_at','fps','frame_count','duration_sec','coordinate_frame','joint_units','orientation_representation','execution_status','is_synthetic','valid_for_paper_result','primitive_source','notes'}
REQUIRED_ARRAYS={'timestamps','fps','full_joint_names','arm_joint_names','left_dex3_joint_names','right_dex3_joint_names','left_phase','right_phase'}
def now():return datetime.now(timezone.utc).isoformat()
def sidecar(path):return path.with_suffix('.metadata.json')
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def remap_joints_by_name(values,source_names,target_names):
 source=[str(x) for x in source_names];target=[str(x) for x in target_names]
 if len(source)!=len(set(source)):raise ValueError('duplicate source joint names')
 missing=[x for x in target if x not in source]
 if missing:raise ValueError(f'missing joints: {missing}')
 return np.asarray(values)[...,[source.index(x) for x in target]]
def validate_behavior_schema(arrays,metadata):
 errors=[];missing=REQUIRED_META-set(metadata)
 if missing:errors.append(f'missing metadata {sorted(missing)}')
 missing_a=REQUIRED_ARRAYS-set(arrays)
 if missing_a:errors.append(f'missing arrays {sorted(missing_a)}')
 if errors:return errors
 if metadata['schema_version']!=SCHEMA:errors.append('schema version mismatch')
 if metadata['behavior_role'] not in ROLES:errors.append('invalid behavior_role')
 if metadata['execution_status'] not in STATUS:errors.append('invalid execution_status')
 n=int(metadata['frame_count']);t=np.asarray(arrays['timestamps'],float)
 if t.shape!=(n,) or not np.isfinite(t).all() or (n>1 and not np.all(np.diff(t)>0)):errors.append('timestamps invalid/non-monotonic')
 for k,v in arrays.items():
  a=np.asarray(v)
  if a.ndim and len(a)==n and np.issubdtype(a.dtype,np.number) and not np.isfinite(a).all():errors.append(f'{k} contains NaN/Inf')
 for prefix in ('target','actual'):
  fq=f'{prefix}_full_qpos';aq=f'{prefix}_arm_qpos'
  if fq in arrays and np.asarray(arrays[fq]).shape!=(n,len(arrays['full_joint_names'])):errors.append(f'{fq} shape')
  if aq in arrays and np.asarray(arrays[aq]).shape!=(n,len(arrays['arm_joint_names'])):errors.append(f'{aq} shape')
 if metadata['behavior_role'] in ('generated_executed','expert_actual') and metadata['execution_status']=='executed' and 'actual_full_qpos' not in arrays:errors.append('executed actual behavior missing actual_full_qpos')
 if metadata['valid_for_paper_result'] and metadata['is_synthetic']:errors.append('synthetic cannot be paper-valid')
 return errors
def save_behavior(path,arrays,metadata):
 path=Path(path);errors=validate_behavior_schema(arrays,metadata)
 if errors:raise ValueError('; '.join(errors))
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 os.replace(tmp,path);mt=sidecar(path).with_suffix('.json.tmp');mt.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n');os.replace(mt,sidecar(path));return path
def load_behavior(path):
 path=Path(path)
 with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 m=json.loads(sidecar(path).read_text());e=validate_behavior_schema(a,m)
 if e:raise ValueError('; '.join(e))
 return a,m
def quat_mul(a,b):
 w,x,y,z=np.moveaxis(a,-1,0);v,s,t,u=np.moveaxis(b,-1,0)
 return np.stack((w*v-x*s-y*t-z*u,w*s+x*v+y*u-z*t,w*t-x*u+y*v+z*s,w*u+x*t-y*s+z*v),-1)
def quat_conj(q):q=np.asarray(q).copy();q[...,1:]*=-1;return q
def compute_behavior_fk(full_qpos,xml):
 import mujoco
 m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);lb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_wrist_yaw_link');rb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link')
 if min(lb,rb)<0:raise ValueError('verified wrist body missing')
 lp=[];rp=[];lq=[];rq=[]
 for q in np.asarray(full_qpos):d.qpos[:]=q;mujoco.mj_forward(m,d);lp.append(d.xpos[lb].copy());rp.append(d.xpos[rb].copy());lq.append(d.xquat[lb].copy());rq.append(d.xquat[rb].copy())
 lp,rp,lq,rq=map(np.asarray,(lp,rp,lq,rq));return {'left_hand_position':lp,'right_hand_position':rp,'left_hand_quaternion':lq,'right_hand_quaternion':rq,'bimanual_midpoint':(lp+rp)/2,'bimanual_relative_position':rp-lp,'bimanual_relative_quaternion':quat_mul(rq,quat_conj(lq))}
def summarize_behavior(arrays,metadata):
 q=arrays.get('actual_arm_qpos',arrays.get('target_arm_qpos'));fps=float(arrays['fps']);out={'behavior_id':metadata['behavior_id'],'role':metadata['behavior_role'],'frames':metadata['frame_count'],'duration_sec':metadata['duration_sec']}
 if q is not None:out.update(arm_path_length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),max_arm_velocity=float(np.abs(np.diff(q,axis=0)*fps).max(initial=0)))
 out['phase_duration_frames']={s:{p:int(np.count_nonzero(arrays[f'{s}_phase']==p)) for p in np.unique(arrays[f'{s}_phase'])} for s in ('left','right')};return out
