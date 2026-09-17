"""Final frozen PAPER_A=A_WRIST and PAPER_B=C_COUPLED TRAIN40 conversion."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import os,time
from .io import read,record,atomic_json
from .practical_finalize import verify_freeze


def plan_case(out,sid,method):
    out=Path(out);area=out/'FINAL_FROZEN';target=area/'paper_planning_cases'/(sid+'_'+method+'.json')
    if target.exists():
        result=read(target)
        if result.get('frozen_manifest')!=record(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'):
            raise RuntimeError('FROZEN_PAPER_CASE_MANIFEST_MISMATCH: '+str(target))
        assert result['source_id']==sid and result['method']==method
        assert all(record(x['path'])==x for x in result['artifacts'])
        return result
    journal=area/'planner_api_calls'/(sid+'_'+method+'.jsonl');journal.parent.mkdir(exist_ok=True)
    os.environ['SOURCE_GUIDED_RRT_AUDIT_LOG']=str(journal);start=time.monotonic()
    if method=='C_COUPLED':
        from .interaction_chain import attempt
        plan=attempt(area,sid,method,scope='TRAIN40_ABC')
    elif method=='A_WRIST':
        from .conversion_attempt import attempt
        from .episode_context import prepare
        from .interaction_chain import preserve_incomplete,summarize_failure
        context=prepare(area,sid,'A','TRAIN40_ABC');preserve_incomplete(context,area,sid)
        plan=attempt(area,sid,'A','TRAIN40_ABC',execute=False,resume=True)
        plan.update(method_key=method,selected_candidate_id='SOURCE_WRIST_REFERENCE')
        if not plan['full_task_plan']:
            plan['first_failure']=summarize_failure(context,sid,Path(plan['connection']) if plan.get('connection') else None)
    else:raise ValueError('Paper method mapping violation')
    from .practical_planning import summarize
    value=dict(source_id=sid,method=method,plan=plan,summary=summarize(plan),planning_seconds=time.monotonic()-start,
        frozen_manifest=record(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'),artifacts=plan.get('artifacts',[]),planner_audit=record(journal) if journal.exists() else None)
    atomic_json(target,value);return value


def run(out,method):
    from .practical_planning import resolve_case
    verify_freeze(out);study=read(out/'PRACTICAL_STUDY.json');ids=study['train_source_ids']
    code='A' if method=='A_WRIST' else 'B';stage='paper_'+code+'_train40';area=out/'FINAL_FROZEN'
    plans={};plan_path=out/('PAPER_'+code+'_PLANNING.json')
    with ProcessPoolExecutor(max_workers=study['fixed_concurrent_planning_workers'],mp_context=get_context('spawn')) as pool:
        pending={pool.submit(plan_case,str(out),sid,method):('FINAL_FROZEN',sid,method) for sid in ids}
        for future in as_completed(pending):
            sid=pending[future][1];plans[sid]=resolve_case(out,future,pending,len(plans),len(ids),stage=stage)
            atomic_json(plan_path,dict(status='IN_PROGRESS',rows=[plans[s] for s in ids if s in plans],method=method))
            atomic_json(out/'WORK_PROGRESS.json',dict(stage=stage,source_id=sid,method=method,substage='Final frozen planning',completed_work=len(plans),remaining_work=40-len(plans)))
    atomic_json(plan_path,dict(status='PASS',rows=[plans[s] for s in ids],method=method))
    verify_freeze(out)
    from .practical_physics import execute_case
    rows=[];artifacts=[plan_path];path=out/('PAPER_'+code+'_RESULTS.json')
    for i,sid in enumerate(ids):
        plan=plans[sid];physical=execute_case(out,area,plan['plan'],stage)
        row=dict(source_id=sid,paper_method='PAPER_'+code,internal_method=method,planning=plan,physical=physical)
        rows.append(row);artifacts.append(area/'physical_case_reports'/(sid+'_'+method+'.json'))
        atomic_json(path,dict(status='IN_PROGRESS',rows=rows,scheduled=40,completed=i+1))
        atomic_json(out/'WORK_PROGRESS.json',dict(stage=stage,source_id=sid,method=method,substage='Final measured physical case complete',completed_work=i+1,remaining_work=39-i))
    verify_freeze(out)
    atomic_json(path,dict(status='PASS',rows=rows,scheduled=40,method=method,
        frozen_manifest=record(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'),diagnostics_included=False))
    return [path,*artifacts]
