"""Separate paired10 reference coupling experiment over frozen converters."""
from pathlib import Path
import fcntl
import os
import subprocess
import time
import numpy as np
from .io import ROOT,OFFLINE,read,record,atomic_json


def freeze(out):
    from .generalization_gate import require
    require(out, 'reference_coupling10')
    from .generation_study import verify_freeze
    generation=verify_freeze(out);split=read(out/'SPLIT_CONTRACT.json')
    indices=np.round(np.linspace(0,34,10)).astype(int).tolist()
    assert len(set(indices))==10 and indices==split['ablation_positions']
    ids=[split['evaluation_source_ids'][i] for i in indices];assert ids==split['ablation_source_ids']
    schedule=[];inputs=[]
    for i,sid in enumerate(ids):
        for c in (('B','B_NO_COUPLING') if i%2==0 else ('B_NO_COUPLING','B')):
            schedule.append(dict(index=len(schedule),source_id=sid,condition=c))
        inputs += [record(p) for p in (out/'source_phase'/sid).iterdir() if p.suffix in ('.json','.npz')]
    result=dict(status='FROZEN_REFERENCE_COUPLING_PAIRED10',schedule=schedule,source_ids=ids,indices=indices,
        dependencies=[record(out/'generation_freeze/CONTRACT.json'),record(out/'DEV35_scenes/MANIFEST.json'),record(ROOT/'tools/contact_coordination/dev_scene_manifest.py'),*inputs],
        planning_wall_budget_s=generation['planning_wall_budget_s'],generator_and_controller='Exact frozen TRAIN40 generation dependencies; enable_coupling selects the corresponding bank only.',
        changed_factors='Only cross-hand predicted object position/rotation residuals; unary source priors, contact bank, joint seeds, constraints and budget unchanged.',
        all_coupling_fit_work_charged_to_both=True,ACT_policy_ablation=False,interpretation='Exploratory reference-level attribution; no learned-policy coupling claim.')
    path=out/'reference_coupling10/CONTRACT.json'
    if path.exists() and read(path)!=result:raise ValueError('Reference freeze changed')
    atomic_json(path,result);return result


def run(out,resume=False):
    from .generalization_gate import require
    require(out, 'reference_coupling10')
    from .generation_study import verify_freeze
    from .episode_context import prepare
    from .conversion_attempt import attempt
    frozen=freeze(out);area=out/'reference_coupling10';path=area/'LEDGER.json';rows=read(path)['rows'] if path.exists() else []
    if rows and not resume:raise ValueError('Existing reference study: use --resume')
    # Do not compete with active dataset physics or training for resources.
    with (out/'TRAIN40_conversion/.generation.lock').open('a+') as shared_lock,(area/'.reference.lock').open('a+') as lock:
        fcntl.flock(shared_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in frozen['schedule']:
            if any(r['index']==item['index'] for r in rows):continue
            verify_freeze(out)
            for dep in frozen['dependencies']:
                if record(dep['path'])!=dep:raise ValueError('Changed reference dependency')
            c,sid=item['condition'],item['source_id'];context=prepare(out,sid,c,'REFERENCE_coupling10')
            args=[OFFLINE,'-m','tools.contact_coordination.conversion_attempt','--run-dir',str(out),'--source-id',sid,'--condition',c,'--scope','REFERENCE_coupling10','--plan-only','--resume']
            log=area/'logs'/f"{item['index']:02d}_{c}_{sid}.log";log.parent.mkdir(exist_ok=True)
            start=time.monotonic();timeout=False;print('REFERENCE_START',item['index']+1,20,c,sid,flush=True)
            with log.open('a') as stream:
                process=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                atomic_json(area/'ACTIVE_PROCESS.json',dict(pid=process.pid,command=args,index=item['index']))
                try:code=process.wait(timeout=frozen['planning_wall_budget_s'])
                except subprocess.TimeoutExpired:
                    timeout=True;process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    code='FIXED_PLANNING_BUDGET_EXHAUSTED'
            elapsed=time.monotonic()-start
            if timeout:row=dict(terminal='NO_PLAN_WITHIN_FIXED_BUDGET',first_failure='PLANNING_WALL_BUDGET',context=str(context),full_task_plan=False,physical_run=False,task_success=None)
            elif code!=0:
                atomic_json(area/'INFRASTRUCTURE_FAILURE.json',dict(item=item,returncode=code,log=record(log)))
                raise RuntimeError('Diagnose reference implementation error: '+str(log))
            else:row=attempt(out,sid,c,'REFERENCE_coupling10',execute=True,resume=True)
            # Canonical reference accounting; preserve the legacy scope label.
            if row.get('terminal') == 'TRAIN_NO_PLAN_WITHIN_BOUNDED_POLICY':
                row=dict(row,original_converter_terminal=row['terminal'],terminal='NO_PLAN_WITHIN_FIXED_BUDGET')
            row=dict(row,**item,planning_process_wall_s=elapsed,planning_log=record(log));rows.append(row)
            atomic_json(path,dict(status='REFERENCE_PAIRED10_RECORDED' if len(rows)==20 else 'REFERENCE_PAIRED10_RUNNING',scheduled=20,completed=len(rows),rows=rows))
            print('REFERENCE_DONE',len(rows),20,c,row['terminal'],flush=True)
    return read(path)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--freeze',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();print(freeze(a.run_dir) if a.freeze else run(a.run_dir,a.resume))
