#!/usr/bin/env python3
"""Resume supported analysis while optional ACT is processed independently."""
from pathlib import Path
import os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_run import DEST,read,atomic_json,file_record,now

def run_stage(script,expected):
    if expected.exists():return
    logdir=DEST/'postprocessing';logdir.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        log=logdir/f'{script}_{time.time_ns()}.log'
        with log.open('w') as stream:
            p=subprocess.run([sys.executable,str(ROOT/'tools'/script)],cwd=ROOT,env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(log.with_suffix('.json'),dict(timestamp=now(),returncode=p.returncode,attempt=attempt,log=file_record(log),expected=str(expected)))
        if p.returncode==0 and expected.exists():return
        print('POSTPROCESS_INFRASTRUCTURE_RETRY',script,attempt,str(log),flush=True)
    raise RuntimeError(f'Postprocessing infrastructure requires repair: {script}; numerical outcomes unchanged')

if __name__=='__main__':
    paper=DEST.parent/'07_paper_artifacts/final'
    while not (DEST/'REFERENCE_PHYSICS_COMPLETE.json').exists():time.sleep(15)
    assert read(DEST/'REFERENCE_PHYSICS_COMPLETE.json')['all_infrastructure_valid']
    run_stage('final_paper_analysis.py',paper/'ANALYSIS_MANIFEST.json')
    run_stage('final_paper_replays.py',paper/'REPLAY_MANIFEST.json')
    while not (DEST/'ACT_BRANCH_RESULT.json').exists():time.sleep(15)
    while not (DEST/'dex3/SOURCE_CLOCK_QUALIFICATION.json').exists():time.sleep(15)
    assert read(DEST/'dex3/SOURCE_CLOCK_QUALIFICATION.json')['status']=='PASS'
    run_stage('final_paper_report.py',DEST/'FINAL_ARTIFACT_VERIFICATION.json')
