#!/usr/bin/env python3
"""Finite, resume-safe independent TRAIN jobs; no method-specific algorithms."""
from pathlib import Path
import sys,subprocess,os,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def worker(case):
    folder=ST5/'train11_orchestration'/case
    for script in ('run_common_train11_position_v5.py','run_common_train_seed_recovery_v6.py'):
        if (TRAIN/case/'SOURCE_POSITION_PASS.json').exists():return dict(case=case,status='QUALIFIED')
        done=folder/(script+'.DONE.json')
        if done.exists() and read(done)['returncode']==0:continue
        cmd=[sys.executable,str(ROOT/'tools'/script),case];start=time.monotonic()
        folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        logfile=folder/(script+'.log')
        print('TRAIN_JOB_START',case,script,flush=True)
        with logfile.open('a') as out:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            atomic_json(folder/'ACTIVE_PROCESS.json',dict(pid=proc.pid,command=cmd,started=datetime.now(timezone.utc).isoformat(),log=str(logfile)))
            code=proc.wait()
        record=dict(case=case,script=script,returncode=code,runtime_s=time.monotonic()-start,log=file_record(logfile))
        atomic_json(done,record);atomic_json(folder/'LAST_PROCESS_RESULT.json',record)
        print('TRAIN_JOB_END',case,script,code,flush=True)
        if code:return dict(case=case,status='INFRASTRUCTURE_RETRY_REQUIRED',result=record)
    return dict(case=case,status='QUALIFIED' if (TRAIN/case/'SOURCE_POSITION_PASS.json').exists() else 'COMMON_RECOVERY_REQUIRED')

def main():
    verified_oracle_contract();smoke=read(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json')
    assert smoke['status']=='SMOKE3_COMMON_EXECUTABLE_POSITION_QUALIFIED'
    for row in smoke['rows']:
        case=row['case'];p=TRAIN/case/'SOURCE_POSITION_PASS.json'
        if not p.exists():
            source=read(V5/case/'SOURCE_POSITION_PASS.json');source['complete_temporal']=row['complete_temporal'];source['preparation_blocked_frames']=[]
            source['preparation_requalification']=file_record(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json');atomic_json(p,source)
    ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids']
    # Episode5 already has live bounded recovery jobs owned by the main agent.
    cases=[f'{mode}_EP{ep:03d}' for ep in ids if ep not in (0,5,24,49) for mode in ('WRIST','INTERACTION')]
    folder=ST5/'train11_orchestration';atomic_json(folder/'CONTRACT.json',dict(cases=cases,parallel_workers=2,
        excluded_active_source_episode=5,algorithm_sequence=['common baseline candidates','common diverse posture-pool recovery'],
        source_manifest=file_record(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')))
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,case) for case in cases]
        for future in as_completed(jobs):
            row=future.result();results.append(row);atomic_json(folder/'PROGRESS.json',dict(results=results,total=len(cases)))
            print('TRAIN_JOB_RESULT',row,flush=True)
    atomic_json(folder/'COMPLETE.json',dict(results=results,not_a_final_pipeline_result=True))

if __name__=='__main__':main()
