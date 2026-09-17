#!/usr/bin/env python3
"""Consume completed frozen position outcomes, keeping invalid stages distinct."""
from pathlib import Path
import concurrent.futures,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/final_single_variable_ab/paper_completion_v1'
PY='/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python'

def run(key):
    folder=D/'full6d'/key;folder.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        with (folder/f'PROCESS_{time.time_ns()}.log').open('w') as log:
            subprocess.run([PY,str(ROOT/'tools/final_paper_full6d_run.py'),key],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        p=folder/'RESULT.json'
        if p.exists() and json.loads(p.read_text())['outcome']!='INFRASTRUCTURE_INVALID':return json.loads(p.read_text())
    return dict(case=dict(key=key),outcome='INFRASTRUCTURE_INVALID')

def main():
    rows=json.loads((D/'CASE_MANIFEST.json').read_text());submitted=set();complete={};futures={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while len(complete)<len(rows):
            for row in rows:
                key=row['key'];p=D/'position'/key/'RESULT.json'
                if key not in submitted and p.exists() and json.loads(p.read_text())['outcome']!='INFRASTRUCTURE_INVALID':
                    futures[pool.submit(run,key)]=key;submitted.add(key)
            for f in list(futures):
                if f.done():
                    key=futures.pop(f);complete[key]=f.result();print('FULL6D_PROGRESS',len(complete),key,complete[key]['outcome'],flush=True)
            if len(complete)<len(rows):time.sleep(10)
    (D/'FULL6D_COHORT_COMPLETE.json').write_text(json.dumps(complete,indent=2))

if __name__=='__main__':main()
