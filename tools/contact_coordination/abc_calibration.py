"""Finite four-recipe development selection over the existing batch workers."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from .io import ROOT,OFFLINE,read,record,atomic_json,atomic_text
from .abc_contract import require
from .abc_audit import csv_write


def source_deviation(plan):
    """Fixed geometric tie-breaker, independent of recipe-weighted solver cost."""
    from .source_phase import mean_pose
    context=Path(plan['context']);sid=plan['source_id']
    phase=read(context/'source_phase'/sid/'PHASE_RECORD.json')
    priors=np.load(context/'source_phase'/sid/'SOURCE_PRIORS.npz')
    prior=mean_pose(priors['inferred_object_from_left'][phase['handoff_sample_indices']])
    selected=read(Path(plan['plan'])/'CONTACT_SELECTION.json')
    predicted=[np.asarray(x) for x in selected['objects'].values()]
    value=float(np.mean([np.sum((x[:3,3]-prior[:3,3])**2) for x in predicted]))
    return dict(mean_squared_handoff_source_position_deviation_m2=value,
                definition='Mean squared selected predicted handoff object position deviation from this episode inferred source handoff prior; fixed one-metre normalization; unknown source orientation excluded',
                prior=prior,source_relation=phase['source_functional_tool_object_relations']['left'])


def fitting_context(out,recipe,fold):
    area=out/'calibration'/recipe/f'fold_{fold}'
    assignment=read(out/'TRAIN40_FOLDS.json')['folds'][fold]
    proposal=read(out/'PARAMETER_SEARCH_SPACE.json')
    contract=dict(recipe=recipe,fold=fold,fitting_source_ids=assignment['fitting'],
        held_out_source_ids=assignment['held_out'],data_derived_statistics={},data_derived_modes=[],
        implementation='Fixed existing embodiment calibration and source-specific priors; no new statistical fitting',
        held_out_empirical_trace_used_for_new_fit=False,historical_Golden_calibration_disclosed=True,
        proposal=record(out/'PARAMETER_SEARCH_SPACE.json'),fold_manifest=record(out/'TRAIN40_FOLDS.json'))
    if (area/'FOLD_FIT_CONTRACT.json').exists():
        if read(area/'FOLD_FIT_CONTRACT.json')!=contract:raise RuntimeError('Changed immutable fold fitting contract')
        return area
    area.mkdir(parents=True,exist_ok=True)
    for rel in ['SPLIT_CONTRACT.json','bootstrap/SPLITS.json','bootstrap/SELECTION.json']:
        target=area/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/rel,target)
    shutil.copytree(out/'target_repair',area/'target_repair',ignore=shutil.ignore_patterns('runtime_check.xml','RUNTIME_CHECK_MODEL.json'))
    for sid in assignment['held_out']:
        shutil.copytree(out/'source_phase'/sid,area/'source_phase'/sid)
    atomic_json(area/'CALIBRATION_PARAMETERS.json',dict(recipe=recipe,values=proposal['recipes'][recipe]))
    atomic_json(area/'FOLD_FIT_CONTRACT.json',contract);return area


def one(out,recipe,fold,sid,method):
    area=fitting_context(out,recipe,fold);case=area/'attempts'/sid/method;case.mkdir(parents=True,exist_ok=True)
    final=case/'RESULT.json'
    if final.exists():
        row=read(final)
        for evidence in row.get('evidence',[]):
            if record(evidence['path'])!=evidence:raise RuntimeError('Changed completed calibration artifact')
        return row
    start=time.monotonic();plan_receipt=case/'PLAN_RESULT.json'
    if not plan_receipt.exists():
        command=[OFFLINE,'-B','-m','tools.contact_coordination.abc_calibration_worker','--run-dir',str(area),
                 '--source-id',sid,'--method',method,'--receipt',str(plan_receipt)]
        atomic_json(case/'INVOCATION.json',dict(command=command,planning_cap_s=900))
        with (case/'planner.log').open('w') as stream:
            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=930,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(case/'PLANNER_PROCESS.json',dict(returncode=process.returncode,wall_s=time.monotonic()-start))
        if process.returncode!=0 or not plan_receipt.exists():
            return dict(recipe=recipe,fold=fold,source_id=sid,method_key=method,terminal='INFRASTRUCTURE_INVALID',
                        full_task=None,plan=False,physical_run=False,first_failure='PLANNER_PROCESS',case=str(case),evidence=[record(case/'PLANNER_PROCESS.json')])
    plan=read(plan_receipt)
    row=dict(recipe=recipe,fold=fold,source_id=sid,method_key=method,terminal=plan['terminal'],
        full_task=False,plan=plan.get('full_task_plan',False),physical_run=False,physical_validity=None,
        first_failure=plan.get('first_failure'),stages={},case=str(case),evidence=[record(plan_receipt)],
        planning_s=plan.get('worker_wall_s'),evidence_channel='OFFICIAL_NOMINAL')
    if row['plan']:
        deviation=source_deviation(plan);row['source_deviation_m2']=deviation['mean_squared_handoff_source_position_deviation_m2']
        atomic_json(case/'SOURCE_DEVIATION.json',deviation);row['evidence'].append(record(case/'SOURCE_DEVIATION.json'))
        from .full_attempt import prepare,launch
        path=case/'PHYSICAL_POINTER.json'
        if path.exists():folder=Path(read(path)['folder'])
        else:
            folder=prepare(out,Path(plan['plan']),f'{recipe}_fold{fold}_{sid}_{method}',method)
            atomic_json(path,dict(folder=str(folder)))
        process=launch(folder,resume=True)
        row['evidence'].append(record(folder/'PROCESS.json'));row['physical_run']=(folder/'event_log.npz').exists()
        if process['returncode']!=0 or not row['physical_run']:
            row.update(terminal='INFRASTRUCTURE_INVALID',full_task=None,first_failure='PHYSICS_PROCESS')
        else:
            from .abc_score import score
            result=read(folder/'ABC_NOMINAL_SCORE.json') if (folder/'ABC_NOMINAL_SCORE.json').exists() else score(folder,Path(plan['context']))
            row.update(terminal=result['terminal'],full_task=result['stages']['FULL_TASK'],physical_validity=result['physical_validity'],
                       first_failure=result['first_failed_stage'],stages=result['stages'],attempt=str(folder))
            row['evidence'] += [record(folder/'ABC_NOMINAL_SCORE.json'),record(folder/'event_log.npz')]
    row['wall_s']=time.monotonic()-start;atomic_json(final,row);return row


def rank(rows,recipe,folds):
    subset=[r for r in rows if r['recipe']==recipe and r['fold'] in folds]
    if len(subset)!=len(folds)*16 or any(r['full_task'] is None for r in subset):
        raise RuntimeError('Comparable complete calibration outcomes required for recipe selection')
    mean={m:float(np.mean([r['full_task'] for r in subset if r['method_key']==m])) for m in ('B_INDEPENDENT','C_COUPLED')}
    intermediate=sum(r['stages'].get('HANDOFF',False)+r['stages'].get('RIGHT_OWNERSHIP',False) for r in subset)
    deviations=[r['source_deviation_m2'] for r in subset if r['plan']]
    return (.5*sum(mean.values()),min(mean.values()),intermediate,sum(r['plan'] for r in subset),
            -float(np.mean(deviations)) if deviations else 0.,
            -sum(r.get('planning_s',0) or 0 for r in subset))


def save_progress(out,rows,phase):
    atomic_json(out/'calibration/LEDGER.json',dict(status='DEVELOPMENT_SELECTION_IN_PROGRESS',phase=phase,completed=len(rows),ceiling=224,rows=rows))
    csv_write(out/'TRAIN40_CONVERTER_SEARCH_RESULTS.csv',[dict(recipe=r['recipe'],fold=r['fold'],source_id=r['source_id'],method=r['method_key'],
        terminal=r['terminal'],plan=r['plan'],physical_run=r['physical_run'],physical_validity=r.get('physical_validity'),
        full_task=r['full_task'],first_failure=r['first_failure'],planning_s=r.get('planning_s'),case=r['case']) for r in rows])
    text=f'CALIBRATION_READY=YES. FINAL_CONFIG_FROZEN=NO. Bounded calibration {phase}: {len(rows)} completed method/source attempts, ceiling224. No final datasets, ACT or new DEV35 outcomes.\n'
    atomic_text(out/'CURRENT_STATUS.md',text);atomic_text(out/'CHATGPT_UPDATE.md',text)


def run(out,resume=False):
    require(out,'bounded_calibration')
    recipes=read(out/'PARAMETER_SEARCH_SPACE.json')['recipe_order'];folds=read(out/'TRAIN40_FOLDS.json')['folds']
    ledger=out/'calibration/LEDGER.json';rows=read(ledger)['rows'] if ledger.exists() else []
    seen={(r['recipe'],r['fold'],r['source_id'],r['method_key']) for r in rows}
    def evaluate(names,fold_numbers,phase):
        for fold in fold_numbers:
            for sid in folds[fold]['held_out']:
                for recipe in names:
                    for method in ('B_INDEPENDENT','C_COUPLED'):
                        key=(recipe,fold,sid,method)
                        if key in seen:continue
                        result=one(out,recipe,fold,sid,method)
                        if result['terminal']=='INFRASTRUCTURE_INVALID':
                            atomic_json(out/'calibration/INFRASTRUCTURE_STOP.json',result)
                            return result
                        rows.append(result);seen.add(key);assert len(rows)<=224
                        save_progress(out,rows,phase)
                        print('CALIBRATION',len(rows),'/',224,recipe,fold,method,sid,result['terminal'],flush=True)
        return None
    error=evaluate(recipes,[0,1],'FIRST_TWO_FOLDS')
    if error:return dict(status='INFRASTRUCTURE_REPAIR_REQUIRED',result=error,continue_downstream=False)
    challenger=max((r for r in recipes if r!='CURRENT'),key=lambda r:(rank(rows,r,[0,1]),-recipes.index(r)))
    atomic_json(out/'calibration/SCREENING_SELECTION.json',dict(current='CURRENT',challenger=challenger,
        ranking={r:rank(rows,r,[0,1]) for r in recipes},selected_from_new_physical_full_task_outcomes=True))
    error=evaluate(['CURRENT',challenger],[2,3,4],'REMAINING_THREE_FOLDS')
    if error:return dict(status='INFRASTRUCTURE_REPAIR_REQUIRED',result=error,continue_downstream=False)
    winner=max(['CURRENT',challenger],key=lambda r:(rank(rows,r,list(range(5))),r=='CURRENT'))
    summary=dict(status='BOUNDED_DEVELOPMENT_SELECTION_COMPLETE',selected_recipe=winner,challenger=challenger,
        rows=len(rows),ceiling=224,rankings={r:rank(rows,r,list(range(5))) for r in ['CURRENT',challenger]},
        final_config_frozen=False,ACT_or_DEV_outcomes_used=False)
    atomic_json(out/'calibration/SELECTION_RESULT.json',summary)
    cvs=[]
    for recipe in recipes:
        for fold in range(5):
            subset=[r for r in rows if r['recipe']==recipe and r['fold']==fold]
            for method in ('B_INDEPENDENT','C_COUPLED'):
                a=[r for r in subset if r['method_key']==method]
                cvs.append(dict(recipe=recipe,fold=fold,method=method,attempts=len(a),plans=sum(r['plan'] for r in a),
                    physical_runs=sum(r['physical_run'] for r in a),successes=sum(bool(r['full_task']) for r in a),
                    status='MEASURED_DEVELOPMENT_SELECTION' if a else 'NOT_SELECTED_FOR_LATER_FOLDS'))
    csv_write(out/'TRAIN40_CONVERTER_CV_RESULTS.csv',cvs);return summary
