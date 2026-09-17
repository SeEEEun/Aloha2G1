#!/usr/bin/env python3
"""Read-only liveness checks and resume of missing owned stage coordinators."""
from pathlib import Path
import os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_io import *
PY='/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

def running(script):
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');stat=(p/'stat').read_text().split()[2]
            if stat not in ('T','Z') and any(Path(a.decode(errors='ignore')).name==script for a in args if a):rows.append(int(p.name))
        except (OSError,ProcessLookupError):pass
    return rows

def launch(script):
    folder=DEST/'supervisor';folder.mkdir(parents=True,exist_ok=True);log=folder/f'{script}_{time.time_ns()}.log'
    with log.open('w') as stream:p=subprocess.Popen([PY,str(ROOT/'tools'/script)],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
    atomic_json(log.with_suffix('.json'),dict(timestamp=now(),pid=p.pid,script=script,reason='Owned coordinator absent; reuse persisted valid outputs. No numeric budget change or scientific-result rerun.'))
    print('RESUME_COORDINATOR',script,p.pid,flush=True)
    return p

if __name__=='__main__':
    children=[]
    while not (DEST/'FINAL_ARTIFACT_VERIFICATION.json').exists():
        for p in children:p.poll()
        items=[
            ('final_paper_full6d_scheduler.py',DEST/'FULL6D_COHORT_COMPLETE.json',['final_paper_full6d_run.py']),
            ('final_paper_complete_action_qualification.py',DEST/'COMPLETE_ACTION_COHORT_COMPLETE.json',[]),
            ('final_paper_reference_physics.py',DEST/'REFERENCE_PHYSICS_COMPLETE.json',['final_paper_physics_isaac.py','final_paper_score_physics.py']),
            ('final_paper_action_audit.py',DEST/'action_dataset_audit/EXACT_ACTION_DATASET_AUDIT.json',[]),
            ('final_paper_act_training_coordinator.py',DEST/'ACT_BRANCH_RESULT.json',['lerobot-train','final_paper_build_act_datasets.py']),
            ('final_paper_postprocess_coordinator.py',DEST/'FINAL_ARTIFACT_VERIFICATION.json',['final_paper_analysis.py','final_paper_replays.py','final_paper_report.py']),
            ('final_paper_progress_monitor.py',DEST/'FINAL_ARTIFACT_VERIFICATION.json',[])]
        complete=list((DEST/'complete_action').glob('*/RESULT.json'))
        if len(complete)==150 and not (DEST/'COMPLETE_ACTION_COHORT_COMPLETE.json').exists():atomic_json(DEST/'COMPLETE_ACTION_COHORT_COMPLETE.json',dict(completed=150,records=[file_record(p) for p in sorted(complete)]))
        for script,done,worker in items:
            if done.exists() or running(script) or any(running(w) for w in worker):continue
            if script=='final_paper_act_training_coordinator.py' and (DEST/'act_training/ACT_TRAINING_COMPLETE.json').exists():continue
            children.append(launch(script))
        if not (DEST/'POSITION_COHORT_COMPLETE.json').exists():
            if not any(running(s) for s in ['final_paper_position_scheduler_v3.py','final_paper_position_scheduler_v2.py','final_paper_position_scheduler.py','final_paper_position_run.py']):children.append(launch('final_paper_position_scheduler.py'))
        atomic_json(DEST/'SUPERVISOR_STATUS.json',dict(timestamp=now(),position=len(list((DEST/'position').glob('*/RESULT.json'))),full6d=len(list((DEST/'full6d').glob('*/RESULT.json'))),complete_action=len(complete),reference=len(list((DEST/'reference_physics').glob('*/RESULT.json'))),numeric_budget_unchanged=True))
        print('PERSISTED_STAGE_COUNTS',read(DEST/'SUPERVISOR_STATUS.json'),flush=True)
        time.sleep(30)
