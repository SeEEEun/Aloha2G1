#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import load_behavior
from align_g1_behaviors import sample
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/behavior_comparison/report'
def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,required=True);p.add_argument('--expert',type=Path,required=True);p.add_argument('--comparison-dir',type=Path,required=True);p.add_argument('--alignment',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args();g,gm=load_behavior(a.generated);e,em=load_behavior(a.expert);result=json.loads((a.comparison_dir/'comparison_metrics.json').read_text());
 with np.load(a.alignment,allow_pickle=False) as z:pairs=z['index_pairs'];method=str(z['method'])
 a.output_dir.mkdir(parents=True,exist_ok=True);watermark='DIAGNOSTIC ONLY / SYNTHETIC TEST' if not result['paper_valid'] else 'ACTUAL VS ACTUAL PRIMARY';summary={'warning':watermark,'comparison_level':result['comparison_level'],'alignment':method,'metrics':result['metrics']};(a.output_dir/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 flat=[]
 def walk(d,p=''):
  for k,v in d.items():
   n=f'{p}.{k}' if p else k
   if isinstance(v,dict):walk(v,n)
   elif isinstance(v,(int,float)) and v is not None:flat.append({'metric':n,'value':v,'unit':'see metric name','paper_valid':result['paper_valid']})
 walk(result['metrics'])
 tables=('per_trial_metrics.csv','aggregate_metrics.csv','exclusion_log.csv','paper_table_main.csv','paper_table_joint_metrics.csv','paper_table_taskspace_metrics.csv','paper_table_timing_metrics.csv')
 for name in tables:
  rows=[] if name=='exclusion_log.csv' else flat
  with (a.output_dir/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=('metric','value','unit','paper_valid') if rows else ('trial','reason'));w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 gi,ei=pairs[:,0],pairs[:,1];armg=sample(g.get('actual_arm_qpos',g.get('target_arm_qpos')),gi);arme=sample(e.get('actual_arm_qpos',e.get('target_arm_qpos')),ei);lhg=sample(g['left_hand_position'],gi);lhe=sample(e['left_hand_position'],ei);rhg=sample(g['right_hand_position'],gi);rhe=sample(e['right_hand_position'],ei);x=np.linspace(0,1,len(pairs))
 def finish(fig,name):fig.suptitle(f'{method} | {watermark}',fontsize=9);fig.tight_layout();fig.savefig(a.output_dir/name,dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,5));ax.plot(x,armg,alpha=.45);ax.plot(x,arme,'--',alpha=.35);ax.set(xlabel='aligned progress',ylabel='joint angle (rad)');finish(fig,'joint_trajectory_overlay.png')
 for side,aa,bb,name in [('left',lhg,lhe,'left_hand_trajectory_3d.png'),('right',rhg,rhe,'right_hand_trajectory_3d.png')]:fig=plt.figure();ax=fig.add_subplot(projection='3d');ax.plot(*aa.T,label='generated');ax.plot(*bb.T,label='expert');ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.legend();finish(fig,name)
 def line(name,ys,ylab):fig,ax=plt.subplots(figsize=(10,4));[ax.plot(x,y,label=l) for l,y in ys];ax.set(xlabel='aligned progress',ylabel=ylab);ax.legend();finish(fig,name)
 line('bimanual_relative_distance.png',[('generated',np.linalg.norm(sample(g['bimanual_relative_position'],gi),axis=1)),('expert',np.linalg.norm(sample(e['bimanual_relative_position'],ei),axis=1))],'distance (m)');line('hand_position_error_over_time.png',[('left',np.linalg.norm(lhg-lhe,axis=1)),('right',np.linalg.norm(rhg-rhe,axis=1))],'position error (m)')
 from compare_vla_generated_vs_g1_expert import qerr
 line('wrist_orientation_error_over_time.png',[('left',qerr(sample(g['left_hand_quaternion'],gi),sample(e['left_hand_quaternion'],ei))),('right',qerr(sample(g['right_hand_quaternion'],gi),sample(e['right_hand_quaternion'],ei)))],'geodesic error (deg)');line('velocity_profile_comparison.png',[('generated',np.linalg.norm(np.gradient(armg,axis=0),axis=1)),('expert',np.linalg.norm(np.gradient(arme,axis=0),axis=1))],'aligned velocity proxy')
 line('phase_timeline.png',[('generated left',np.array([{'OPEN':0,'PREGRASP':1,'GRASP':2,'HOLD':3,'RELEASE':4}[p] for p in g['left_phase'][np.rint(gi).astype(int)]])),('expert left',np.array([{'OPEN':0,'PREGRASP':1,'GRASP':2,'HOLD':3,'RELEASE':4}.get(p,-1) for p in e['left_phase'][np.rint(ei).astype(int)]]))],'phase code')
 fig,ax=plt.subplots();ax.plot(pairs[:,0],pairs[:,1]);ax.set(xlabel='generated frame',ylabel='expert frame');finish(fig,'dtw_alignment_path.png')
 fig,ax=plt.subplots(figsize=(10,4));vals=list(result['metrics']['per_joint_rmse'].values());ax.bar(range(len(vals)),vals);ax.set(xlabel='arm joint index',ylabel='RMSE (rad)');finish(fig,'per_joint_rmse.png')
 fig,ax=plt.subplots();ax.boxplot([vals]);ax.set(ylabel='per-joint RMSE (rad)',xticklabels=['synthetic expert']);finish(fig,'expert_trial_distribution.png');print(a.output_dir);return 0
if __name__=='__main__':raise SystemExit(main())
