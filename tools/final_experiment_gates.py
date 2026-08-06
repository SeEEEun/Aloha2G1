#!/usr/bin/env python3
"""Fail-closed gates for the offline/open-loop final robot experiments."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/final_robot_experiments/gates'
NAMES=('A0_ALOHA_PREFLIGHT_PASS','A1_ALOHA_SHORT_NO_OBJECT_PASS','A2_ALOHA_SEGMENTS_NO_OBJECT_PASS','A3_ALOHA_FULL_NO_OBJECT_PASS','A4_ALOHA_SOURCE_RESULT','B0_G1_LAYOUT_MEASURED','B1_G1_TASK_FRAME_REGISTERED','B2_G1_TASK_VALID_TRAJECTORY_PASS','C0_DEX3_REAL_PRIMITIVES_PASS','C1_G1_OBJECT_PHYSICS_PASS','D0_G1_READONLY_STATE_PASS','D1_G1_NO_OBJECT_SHORT_PASS','D2_G1_NO_OBJECT_FULL_PASS','D3_G1_GENERATED_PILOT_PASS','D4_G1_GENERATED_TRIALS_COMPLETE','E0_G1_EXPERT_SETUP_PASS','E1_G1_EXPERT_TRIALS_COMPLETE','F0_PRIMARY_ACTUAL_VS_ACTUAL_AVAILABLE')
REQUIRES={'A1_ALOHA_SHORT_NO_OBJECT_PASS':['A0_ALOHA_PREFLIGHT_PASS'],'A2_ALOHA_SEGMENTS_NO_OBJECT_PASS':['A1_ALOHA_SHORT_NO_OBJECT_PASS'],'A3_ALOHA_FULL_NO_OBJECT_PASS':['A2_ALOHA_SEGMENTS_NO_OBJECT_PASS'],'A4_ALOHA_SOURCE_RESULT':['A3_ALOHA_FULL_NO_OBJECT_PASS'],'B0_G1_LAYOUT_MEASURED':['A4_ALOHA_SOURCE_RESULT'],'B1_G1_TASK_FRAME_REGISTERED':['B0_G1_LAYOUT_MEASURED'],'B2_G1_TASK_VALID_TRAJECTORY_PASS':['B1_G1_TASK_FRAME_REGISTERED'],'C0_DEX3_REAL_PRIMITIVES_PASS':['B2_G1_TASK_VALID_TRAJECTORY_PASS'],'C1_G1_OBJECT_PHYSICS_PASS':['C0_DEX3_REAL_PRIMITIVES_PASS'],'D0_G1_READONLY_STATE_PASS':['C1_G1_OBJECT_PHYSICS_PASS'],'D1_G1_NO_OBJECT_SHORT_PASS':['D0_G1_READONLY_STATE_PASS'],'D2_G1_NO_OBJECT_FULL_PASS':['D1_G1_NO_OBJECT_SHORT_PASS'],'D3_G1_GENERATED_PILOT_PASS':['D2_G1_NO_OBJECT_FULL_PASS'],'D4_G1_GENERATED_TRIALS_COMPLETE':['D3_G1_GENERATED_PILOT_PASS'],'E0_G1_EXPERT_SETUP_PASS':['D4_G1_GENERATED_TRIALS_COMPLETE'],'E1_G1_EXPERT_TRIALS_COMPLETE':['E0_G1_EXPERT_SETUP_PASS'],'F0_PRIMARY_ACTUAL_VS_ACTUAL_AVAILABLE':['D4_G1_GENERATED_TRIALS_COMPLETE','E1_G1_EXPERT_TRIALS_COMPLETE']}
def git():
 try:return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
 except Exception:return 'NOT_A_GIT_WORKTREE'
def record(name,status='BLOCKED',operator='UNASSIGNED',evidence=None,metrics=None,notes='',video=None,next_stage=None,input_hashes=None):
 if name not in NAMES:raise ValueError(name)
 evidence=list(evidence or []);video=list(video or [])
 if status=='PASS':
  if not evidence:raise ValueError('PASS requires evidence files')
  missing=[x for x in evidence if not Path(x).exists()]
  if missing:raise ValueError(f'PASS evidence missing: {missing}')
  if not input_hashes:raise ValueError('PASS requires input hashes')
  for req in REQUIRES.get(name,[]):
   p=OUT/f'{req}.json'
   if not p.exists() or json.loads(p.read_text()).get('status')!='PASS':raise ValueError(f'prerequisite not PASS: {req}')
 payload={'status':status,'date':datetime.now(timezone.utc).isoformat(),'operator':operator,'git_commit':git(),'input_hashes':input_hashes or {},'evidence_files':evidence,'video_files':video,'metrics':metrics or {},'human_notes':notes,'next_permitted_stage':next_stage or 'NONE_WHILE_BLOCKED'}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/f'{name}.json').write_text(json.dumps(payload,indent=2)+'\n');return payload
def source_allows_g1():
 p=OUT/'A4_ALOHA_SOURCE_RESULT.json'
 if not p.exists():return False
 d=json.loads(p.read_text());return d.get('status')=='PASS' and d.get('metrics',{}).get('classification') in ('SOURCE_ACTION_VALID','ARM_VALID_CONTACT_FAILED')
def initialize():
 for name in NAMES:
  p=OUT/f'{name}.json'
  if not p.exists():record(name,notes='Template initialized BLOCKED; no hardware/simulation result inferred.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--initialize-blocked',action='store_true');p.add_argument('--check');a=p.parse_args()
 if a.initialize_blocked:initialize();print(OUT);return 0
 if a.check:
  q=OUT/f'{a.check}.json';print(q.read_text() if q.exists() else 'MISSING');return 0 if q.exists() else 2
 p.error('choose --initialize-blocked or --check')
if __name__=='__main__':raise SystemExit(main())
