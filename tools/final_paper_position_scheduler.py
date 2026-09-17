#!/usr/bin/env python3
"""Persist each frozen result; retry infrastructure only, never scientific loss."""
from pathlib import Path
import concurrent.futures, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/final_single_variable_ab/paper_completion_v1'
PY='/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python'

def run(row):
    folder=DEST/'position'/row['key'];folder.mkdir(parents=True,exist_ok=True)
    result=folder/'RESULT.json'
    if result.exists() and json.loads(result.read_text())['outcome']!='INFRASTRUCTURE_INVALID':return json.loads(result.read_text())
    for attempt in range(3):
        logpath=folder/f'PROCESS_{time.time_ns()}.log'
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with logpath.open('w') as log:
            rc=subprocess.run([PY,str(ROOT/'tools/final_paper_position_run.py'),row['key']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if result.exists():
            r=json.loads(result.read_text())
            if r['outcome']!='INFRASTRUCTURE_INVALID':return r
            archive=folder/f'INFRASTRUCTURE_ATTEMPT_{time.time_ns()}.json'
            archive.write_text(result.read_text())
        print('INFRASTRUCTURE_RETRY',row['key'],attempt,rc,str(logpath),flush=True)
    return dict(case=row,outcome='INFRASTRUCTURE_INVALID',reason='Three bounded process attempts exhausted; inspect logs and repair commonly')

def main():
    rows=json.loads((DEST/'CASE_MANIFEST.json').read_text());results=[]
    # Entire TRAIN cohort is processed before DEV; no algorithm changes between.
    for group in ('TRAIN40','DEV35'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(run,r):r for r in rows if r['group']==group}
            for future in concurrent.futures.as_completed(futures):
                r=future.result();results.append(r)
                with (DEST/'EPISODE_COMPLETION_LOG.jsonl').open('a') as log:log.write(json.dumps(dict(timestamp=time.time(),key=r['case']['key'],outcome=r['outcome']))+'\n')
                print('COHORT_PROGRESS',len(results),'/150',r['case']['key'],r['outcome'],flush=True)
    (DEST/'POSITION_COHORT_COMPLETE.json').write_text(json.dumps(dict(results=results,all_infrastructure_valid=all(r['outcome']!='INFRASTRUCTURE_INVALID' for r in results)),indent=2))

if __name__=='__main__':main()
