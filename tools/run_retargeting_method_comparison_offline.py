#!/usr/bin/env python3
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path('/home/jbnu/aloha_g1_dataset');PY=sys.executable;T=ROOT/'tools';OUT=ROOT/'outputs/retargeting_method_comparison';METHODS=('relative_temporal_proposed','relative_framewise','relative_no_workspace_scale')
def run(*a):subprocess.run([str(x) for x in a],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();[p.add_argument('--'+x,action='store_true') for x in ('audit','generate','validate','render','isaaclab','report','all')];a=p.parse_args();all_=a.all or not any(vars(a).values())
 if all_ or a.generate:run(PY,T/'generate_retargeting_method_comparison.py')
 if all_ or a.audit or a.validate or a.render or a.report:run(PY,T/'evaluate_retargeting_method_comparison.py',*(['--render'] if all_ or a.render else ['--no-render']))
 if all_ or a.isaaclab:
  for m in METHODS:
   run('/home/jbnu/IsaacLab-3-beta/isaaclab.sh','-p',ROOT/'isaaclab_magsafe_fixed_scene/replay_g1_arm_dex3_offline.py','--input',OUT/'trajectories'/m/'full_trajectory_placeholder_dex3.npz','--output-dir',OUT/'isaaclab'/m,'--device','cpu','--headless')
  run(PY,T/'evaluate_retargeting_method_comparison.py','--no-render')
 if all_ or a.audit:run(PY,T/'validate_vla_action_on_real_aloha.py','--dry-run','--inspect','--no-object')
 print('\nRETARGETING METHOD COMPARISON COMPLETE\nNO REAL G1 WAS USED\nDEX3 PLACEHOLDER RESULTS ARE DIAGNOSTIC ONLY\nOPEN THE DASHBOARD:\n'+str(OUT/'report/index.html'));return 0
if __name__=='__main__':raise SystemExit(main())
