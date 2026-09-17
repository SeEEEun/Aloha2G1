"""Planning-only four-recipe jobs and the predeclared stop decision."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import os,time,csv,json,traceback
import numpy as np
from .io import read,record,atomic_json,atomic_text
from .practical_study import METHODS,assert_backend


def csv_file(path,rows):
    import io
    stream=io.StringIO();fields=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    atomic_text(path,stream.getvalue())


def summarize(plan):
    sid=plan['source_id'];ik=0;valid=0;generated=plan.get('generated_grasp_candidates',1)
    attempts=plan.get('attempts',[dict(context=plan['context'])])
    for branch in attempts:
        p=Path(branch['context'])/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json'
        if p.exists():
            for phase in read(p)['phases']:
                ik+=sum(bool(c.get('goal_satisfied')) for c in phase.get('candidates',[]))
                valid+=sum(bool(c.get('admissible')) for c in phase.get('candidates',[]))
    failure=plan.get('first_failure');label=failure.get('cause') if isinstance(failure,dict) else failure
    source=None
    if plan['full_task_plan']:
        from .abc_calibration import source_deviation
        source=source_deviation(plan)['mean_squared_handoff_source_position_deviation_m2']
    return dict(GENERATED_GRASP_TARGETS=generated,IK_VALID=ik,VALID_CANDIDATES=valid,
        candidate_count_unit='IK endpoint solutions across actually visited acquisition branches/phases; generated_grasp_targets separately reported',
        VALID_TARGET=ik>0,COMPLETE_PLAN=bool(plan['full_task_plan']),
        **{k:int(not plan['full_task_plan'] and label==k) for k in ('NO_IK','NO_VALID_CANDIDATE','NO_CONNECTING_PATH','NO_COMPLETE_CHAIN')},
        first_causal_failure=failure,source_deviation_m2=source)


def worker(out,recipe,sid,method):
    out=Path(out);area=out/'recipes'/recipe
    journal=area/'planner_api_calls'/(sid+'_'+method+'.jsonl');journal.parent.mkdir(exist_ok=True)
    os.environ['SOURCE_GUIDED_RRT_AUDIT_LOG']=str(journal)
    result_path=area/'case_receipts'/(sid+'_'+method+'.json')
    from .scientific_cache import key as cache_key
    scientific_key=cache_key(area,sid,'practical_plan_case',dict(method=method,recipe=recipe))
    if result_path.exists():
        value=read(result_path)
        if value.get('scientific_key')==scientific_key:
            assert all(record(x['path'])==x for x in value['artifacts'])
            return value
        archive=area/'INVALIDATED_CASE_RECEIPTS'/(sid+'_'+method+'_'+record(result_path)['sha256'][:12]+'.json')
        atomic_json(archive,value)
    started=time.monotonic()
    from .interaction_chain import attempt
    plan=attempt(area,sid,method,scope='TRAIN_calibration')
    result=dict(source_id=sid,method=method,recipe=recipe,scientific_key=scientific_key,plan=plan,summary=summarize(plan),
        planning_seconds=time.monotonic()-started,physics_executed=False,
        artifacts=[record(plan['receipt'])],planner_audit=record(journal) if journal.exists() else None)
    atomic_json(result_path,result);return result


def resolve_case(out,future,pending,completed,scheduled,stage='planning_sweep'):
    """Publish infrastructure exceptions immediately; preserve running work."""
    try:return future.result()
    except Exception as error:
        recipe,sid,method=pending[future]
        failure=dict(status='INFRASTRUCTURE_FAILURE',recipe=recipe,source_id=sid,method=method,
            exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        path=out/'INFRASTRUCTURE_FAILURES'/(recipe+'_'+sid+'_'+method+'_'+str(time.time_ns())+'.json')
        atomic_json(path,failure)
        cancelled=sum(f.cancel() for f in pending if f is not future)
        atomic_json(out/'WORK_PROGRESS.json',dict(stage=stage,recipe=recipe,source_id=sid,method=method,
            completed_work=completed,remaining_work=scheduled-completed,
            substage='Infrastructure exception recorded; queued jobs cancelled; draining already running jobs',
            infrastructure_failure=record(path),cancelled_pending_jobs=cancelled))
        print(failure['traceback'],flush=True)
        raise


def run(out):
    assert_backend(out);study=read(out/'PRACTICAL_STUDY.json');ids=study['train_source_ids'];recipes=study['recipes']
    jobs=[(recipe,sid,method) for recipe in recipes for sid in ids for method in METHODS]
    rows={};path=out/'PLANNING_SWEEP.json'
    atomic_json(out/'WORK_PROGRESS.json',dict(stage='planning_sweep',completed_work=0,remaining_work=len(jobs),
        substage='Verify case dependencies and execute incomplete planning cases'))
    with ProcessPoolExecutor(max_workers=study['fixed_concurrent_planning_workers'],mp_context=get_context('spawn')) as pool:
        pending={pool.submit(worker,str(out),*job):job for job in jobs}
        for future in as_completed(pending):
            key=pending[future];rows[key]=resolve_case(out,future,pending,len(rows),len(jobs))
            ordered=[rows[k] for k in jobs if k in rows]
            atomic_json(path,dict(status='IN_PROGRESS',rows=ordered,completed=len(rows),scheduled=len(jobs)))
            atomic_json(out/'WORK_PROGRESS.json',dict(stage='planning_sweep',recipe=key[0],source_id=key[1],method=key[2],
                completed_work=len(rows),remaining_work=len(jobs)-len(rows),substage='Completed immutable planning case'))
    ordered=[rows[k] for k in jobs]
    assert_backend(out)
    atomic_json(path,dict(status='PASS',rows=ordered,scheduled=320,physics_executed=False,all_four_recipes_complete=True))
    csv_path=out/'PLANNING_SWEEP.csv';csv_file(csv_path,[dict(source_id=r['source_id'],method=r['method'],recipe=r['recipe'],planning_seconds=r['planning_seconds'],**r['summary']) for r in ordered])
    return [path,csv_path,*[out/'recipes'/r/'case_receipts'/(sid+'_'+m+'.json') for r,sid,m in jobs]]


def rank(out):
    study=read(out/'PRACTICAL_STUDY.json');rows=read(out/'PLANNING_SWEEP.json')['rows'];ranked=[]
    for complexity,recipe in enumerate(study['recipes']):
        group=[r for r in rows if r['recipe']==recipe]
        complete={m:sum(r['summary']['COMPLETE_PLAN'] for r in group if r['method']==m) for m in METHODS}
        deviations=[r['summary']['source_deviation_m2'] for r in group if r['summary']['source_deviation_m2'] is not None]
        ranked.append(dict(recipe=recipe,complete=complete,score=sum(complete.values())/80.,weaker_method=min(complete.values()),
            no_ik=sum(r['summary']['NO_IK'] for r in group),no_complete_chain=sum(r['summary']['NO_COMPLETE_CHAIN'] for r in group),
            source_deviation=float(np.mean(deviations)) if deviations else None,planning_seconds=sum(r['planning_seconds'] for r in group),complexity=complexity))
    key=lambda r:(-r['score'],-r['weaker_method'],r['no_ik'],r['no_complete_chain'],r['source_deviation'] if r['source_deviation'] is not None else float('inf'),r['planning_seconds'],r['complexity'])
    ranked.sort(key=key);default=next(r for r in ranked if r['recipe']=='CURRENT_DEFAULT')
    old={m:{r['source_id'] for r in rows if r['recipe']=='CURRENT_DEFAULT' and r['method']==m and r['summary']['COMPLETE_PLAN']} for m in METHODS}
    for r in ranked:
        new={m:{x['source_id'] for x in rows if x['recipe']==r['recipe'] and x['method']==m and x['summary']['COMPLETE_PLAN']} for m in METHODS}
        gain=set().union(*(new[m]-old[m] for m in METHODS));average_gain=(r['score']-default['score'])*40
        r.update(new_distinct_sources=sorted(gain),average_net_gain=average_gain,
            meaningful_improvement=average_gain>=2.-1e-9 and len(gain)>=2)
    improved=any(r['meaningful_improvement'] for r in ranked)
    top=[ranked[0]['recipe']] if improved else []
    if improved and ranked[0]['score']-ranked[1]['score']<=.0125+1e-9 and ranked[0]['weaker_method']-ranked[1]['weaker_method']<=1:top.append(ranked[1]['recipe'])
    path=out/'RECIPE_RANKING.json';atomic_json(path,dict(status='PASS',ranked=ranked,planning_improved=improved,top_recipes=top,selection_uses_method_gap=False))
    md='# Four-recipe planning calibration\n\n| Recipe | B complete /40 | C complete /40 | Shared score | New distinct sources | Meaningful improvement |\n|---|---:|---:|---:|---:|---|\n'
    for r in sorted(ranked,key=lambda r:r['complexity']):md+=f"| {r['recipe']} | {r['complete']['B_INDEPENDENT']} | {r['complete']['C_COUPLED']} | {r['score']:.4f} | {len(r['new_distinct_sources'])} | {r['meaningful_improvement']} |\n"
    md+='\nTop recipes for predeclared physics: '+(', '.join(top) if top else 'NONE; planning improvement gate closed')+'\n'
    report=out/'PLANNING_RECIPE_COMPARISON.md';atomic_text(report,md)
    artifacts=[path,report]
    if not improved:
        from collections import Counter
        causes=Counter()
        for r in rows:
            f=r['summary']['first_causal_failure']
            if f:causes[(f.get('cause','UNKNOWN'),f.get('phase','UNKNOWN'))]+=1
        diagnostic=out/'PLANNING_CALIBRATION_INSUFFICIENT.md'
        text=md+'\nNo recipe meets the predeclared requirement of two net additional plans per method on average and two distinct newly covered sources. Expensive physics and final paper evaluation were not started. No further recipes or architecture changes are authorized by this run.\n\n'
        text+='First causal failure counts across320 scheduled recipe/method/source cases:\n\n'+''.join(f'- {cause} at {phase}: {n}\n' for (cause,phase),n in causes.most_common())
        text+='\nThe swept dimensions were preparation height, grasp translation radius, grasp orientation freedom, acquisition IK posture weight, shallow incidental penetration and object-displacement tolerance. The displacement tolerance affects physical acceptance only and cannot repair NO_IK. If NO_IK dominates at preparation, the fixed endpoint/candidate reachable region remains the indicated bottleneck; this planning sweep does not prove geometric impossibility or justify changing the representation.\n\nDEV35 STARTED: NO\nACT STARTED: NO\n'
        atomic_text(diagnostic,text);artifacts.append(diagnostic)
    return artifacts
