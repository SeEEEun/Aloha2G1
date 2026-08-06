#!/usr/bin/env python3
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path('/home/jbnu/aloha_g1_dataset');PY=sys.executable;T=ROOT/'tools';GEN=ROOT/'outputs/behavior_comparison/generated/episode49_vla_generated_target.npz';SYN=ROOT/'outputs/behavior_comparison/synthetic';MAN=ROOT/'configs/g1_behavior_comparison_manifest.json'
def run(*x):subprocess.run([PY,*map(str,x)],check=True,cwd=ROOT)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('prepare-generated','synthetic-test','compare','report','all-offline'),default='all-offline');p.add_argument('--generated-executed',type=Path);p.add_argument('--expert-dir',type=Path);a=p.parse_args()
 if a.mode in ('prepare-generated','all-offline'):run(T/'export_vla_generated_g1_behavior.py');run(T/'create_g1_behavior_comparison_manifest.py');run(T/'annotate_g1_behavior_events.py','--behavior',GEN)
 if a.mode in ('synthetic-test','all-offline'):run(T/'generate_synthetic_g1_expert_for_testing.py')
 if a.generated_executed and a.expert_dir:
  experts=sorted(a.expert_dir.glob('*.npz'));generated=a.generated_executed
 else:experts=[SYN/'identical.npz',SYN/'time_warped.npz',SYN/'joint_offset.npz',SYN/'hand_position_offset.npz',SYN/'quaternion_sign_flip.npz',SYN/'different_fps.npz',SYN/'missing_event.npz'];generated=GEN
 if a.mode in ('compare','report','all-offline'):
  comp=[]
  for expert in experts:
   for method in ('raw_time','normalized_time','dtw_hand','phase_aligned'):
    od=ROOT/f'outputs/behavior_comparison/comparisons/{expert.stem}/{method}';al=od/'alignment.npz'
    try:run(T/'align_g1_behaviors.py','--generated',generated,'--expert',expert,'--method',method,'--output',al);run(T/'compare_vla_generated_vs_g1_expert.py','--generated',generated,'--expert',expert,'--alignment',al,'--manifest',MAN,'--output-dir',od);comp.append(od)
    except subprocess.CalledProcessError:
     if method!='phase_aligned':raise
  report_expert=experts[0];od=ROOT/f'outputs/behavior_comparison/comparisons/{report_expert.stem}/dtw_hand'
  run(T/'generate_g1_behavior_comparison_report.py','--generated',generated,'--expert',report_expert,'--comparison-dir',od,'--alignment',od/'alignment.npz')
  run(T/'aggregate_g1_expert_comparisons.py','--comparisons',*comp,'--output-dir',ROOT/'outputs/behavior_comparison/report')
 if a.mode in ('synthetic-test','all-offline'):run(T/'diagnose_g1_generated_collision_events.py');run(ROOT/'tests/test_g1_behavior_comparison.py')
 print('\nOFFLINE PREPARATION COMPLETE\nNO REAL G1 DATA WAS USED\nSYNTHETIC COMPARISON IS NOT A PAPER RESULT');return 0
if __name__=='__main__':raise SystemExit(main())
