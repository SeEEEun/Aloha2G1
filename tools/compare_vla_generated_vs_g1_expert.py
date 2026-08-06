#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import load_behavior,remap_joints_by_name
from align_g1_behaviors import sample
def qerr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True);return 2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=1)),-1,1))*180/np.pi
def pathlen(x):return float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())
def actual_or_target(a,m,prefix):
 k=f'actual_{prefix}'
 if k in a:return a[k]
 k=f'target_{prefix}'
 if k in a:return a[k]
 raise ValueError(f'no actual or target {prefix}')
def compute(g,gm,e,em,pairs):
 gi,ei=pairs[:,0],pairs[:,1];ga=sample(actual_or_target(g,gm,'arm_qpos'),gi);ea=sample(remap_joints_by_name(actual_or_target(e,em,'arm_qpos'),e['arm_joint_names'],g['arm_joint_names']),ei);de=ga-ea;fps=min(float(g['fps']),float(e['fps']));gv=np.gradient(ga,1/fps,axis=0);ev=np.gradient(ea,1/fps,axis=0);gac=np.gradient(gv,1/fps,axis=0);eac=np.gradient(ev,1/fps,axis=0)
 out={'joint_mae':float(np.mean(np.abs(de))),'joint_rmse':float(np.sqrt(np.mean(de**2))),'joint_max_error':float(np.max(np.abs(de))),'per_joint_rmse':dict(zip(g['arm_joint_names'].tolist(),np.sqrt(np.mean(de**2,axis=0)).tolist())),'mean_abs_velocity_difference':float(np.mean(np.abs(gv-ev))),'mean_abs_acceleration_difference':float(np.mean(np.abs(gac-eac))),'generated_arm_path_length':pathlen(ga),'expert_arm_path_length':pathlen(ea),'arm_path_length_difference':abs(pathlen(ga)-pathlen(ea))}
 for side in ('left','right'):
  gp=sample(g[f'{side}_hand_position'],gi);ep=sample(e[f'{side}_hand_position'],ei);err=np.linalg.norm(gp-ep,axis=1);oq=qerr(sample(g[f'{side}_hand_quaternion'],gi),sample(e[f'{side}_hand_quaternion'],ei));out.update({f'{side}_hand_position_rmse_m':float(np.sqrt(np.mean(err**2))),f'{side}_hand_position_p95_m':float(np.percentile(err,95)),f'{side}_path_length_difference_m':abs(pathlen(gp)-pathlen(ep)),f'{side}_orientation_mean_deg':float(np.mean(oq)),f'{side}_orientation_median_deg':float(np.median(oq)),f'{side}_orientation_p95_deg':float(np.percentile(oq,95))})
 gmpos=sample(g['bimanual_midpoint'],gi);empos=sample(e['bimanual_midpoint'],ei);gr=sample(g['bimanual_relative_position'],gi);er=sample(e['bimanual_relative_position'],ei);gd=np.linalg.norm(gr,axis=1);ed=np.linalg.norm(er,axis=1);out.update(bimanual_midpoint_rmse_m=float(np.sqrt(np.mean((gmpos-empos)**2))),relative_hand_position_rmse_m=float(np.sqrt(np.mean((gr-er)**2))),inter_hand_distance_mae_m=float(np.mean(np.abs(gd-ed))),relative_orientation_mean_deg=float(np.mean(qerr(sample(g['bimanual_relative_quaternion'],gi),sample(e['bimanual_relative_quaternion'],ei)))),approach_separation_correlation=float(np.corrcoef(np.gradient(gd),np.gradient(ed))[0,1]) if np.std(gd)*np.std(ed)>0 else 1.0)
 for side in ('left','right'):
  try:gq=sample(actual_or_target(g,gm,f'{side}_dex3_qpos'),gi);eq=sample(remap_joints_by_name(actual_or_target(e,em,f'{side}_dex3_qpos'),e[f'{side}_dex3_joint_names'],g[f'{side}_dex3_joint_names']),ei);out[f'{side}_dex3_qpos_rmse']=float(np.sqrt(np.mean((gq-eq)**2)))
  except ValueError:out[f'{side}_dex3_qpos_rmse']=None
  gp=g[f'{side}_phase'][np.rint(gi).astype(int)];ep=e[f'{side}_phase'][np.rint(ei).astype(int)];out[f'{side}_phase_agreement']=float(np.mean(gp==ep))
 def et(arr,name):
  if 'event_names' not in arr:return None
  idx=np.flatnonzero(arr['event_names']==name);return float(arr['event_timestamps'][idx[0]]) if len(idx) else None
 timing={}
 for s in ('left','right'):
  for ev in ('grasp_start','release_start'):
   x,y=et(g,f'{s}_{ev}'),et(e,f'{s}_{ev}');timing[f'{s}_{ev}_difference_sec']=None if x is None or y is None else abs(x-y)
 timing['task_duration_difference_sec']=abs(float(g['timestamps'][-1])-float(e['timestamps'][-1]));out['timing']=timing;return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,required=True);p.add_argument('--expert',type=Path,required=True);p.add_argument('--alignment',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();g,gm=load_behavior(a.generated);e,em=load_behavior(a.expert)
 if gm.get('left_right_swapped') or em.get('left_right_swapped'):raise RuntimeError('LEFT_RIGHT_SWAP_DETECTED')
 with np.load(a.alignment,allow_pickle=False) as z:pairs=z['index_pairs'];method=str(z['method'])
 primary=(gm['behavior_role']=='generated_executed' and em['behavior_role']=='expert_actual' and gm['execution_status']=='executed' and em['execution_status']=='executed' and not gm['is_synthetic'] and not em['is_synthetic'] and 'actual_full_qpos' in g and 'actual_full_qpos' in e)
 level='actual_vs_actual_primary' if primary else 'target_vs_actual_diagnostic';metrics=compute(g,gm,e,em,pairs);result={'status':'PRIMARY_RESULT_AVAILABLE' if primary else 'PRIMARY_RESULT_NOT_AVAILABLE','comparison_level':level,'warning':None if primary else 'DIAGNOSTIC ONLY - NOT A PAPER EXPERIMENT RESULT','paper_valid':primary,'alignment_method':method,'generated':str(a.generated),'expert':str(a.expert),'metrics':metrics};a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'comparison_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
