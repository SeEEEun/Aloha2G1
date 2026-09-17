"""Targeted-repair gates over the existing conversion and study runners."""
from pathlib import Path
import os,subprocess,time,json
from .io import ROOT,OFFLINE,read,record,atomic_json,atomic_text


def plan(out,sid,condition,scope,receipt,resume):
    from .episode_context import prepare
    if receipt.exists():
        if not resume:raise FileExistsError(receipt)
        saved=read(receipt)
        if saved.get('context') and Path(saved['context'])!=prepare(out,sid,condition,scope):
            raise ValueError('Scientific dependencies changed; preserve the old attempt and record a new version')
        return saved
    receipt.parent.mkdir(parents=True,exist_ok=True)
    args=[OFFLINE,'-B','-m','tools.contact_coordination.targeted_batch_worker','--run-dir',str(out),
          '--source-id',sid,'--condition',condition,'--scope',scope,'--receipt',str(receipt)]
    with receipt.with_suffix('.log').open('w') as stream:
        p=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(receipt.with_name('ACTIVE_PROCESS.json'),dict(pid=p.pid,command=args,time=time.time()))
        p.wait(timeout=930)
    if p.returncode or not receipt.exists():raise RuntimeError('Inspect bounded batch process log: '+str(receipt.with_suffix('.log')))
    return read(receipt)


def prototype(out,resume):
    propagation=read(out/'target_repair/RIGHT_RECEIVING_INTEGRATION_RESULT.json')
    if propagation['SOURCE_RIGHT_RELATION_PROPAGATES']!='YES':raise ValueError('Receiving sensitivity gate unmet')
    for item in propagation['code'].values():
        if record(item['path'])!=item:raise ValueError('Receiving implementation changed after sensitivity test')
    if read(out/'target_repair/CACHE_BOUNDARY_REGRESSION.json')['status']!='PASS':raise ValueError('Cache regression gate unmet')
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    receipt=out/read(out/'TARGETED_REPAIR_CONTRACT.json')['golden_plan_receipt']
    row=plan(out,sid,'B','TRAIN_pilot',receipt,resume)
    if not row.get('full_task_plan'):return dict(status='GOLDEN_PLAN_NOT_DEMONSTRATED',continue_downstream=False,row=row)
    equivalent=out/'golden_batch/CURRENT_PHYSICAL_EQUIVALENCE.json'
    if equivalent.exists():
        verified=read(equivalent)
        if verified['new_plan']==row['plan']:
            for item in verified['artifacts']+verified['runtime_dependencies']:
                if record(item['path'])!=item:raise ValueError('Changed physical equivalence dependency')
            if verified['status']!='EXACT_COMMAND_AND_RUNTIME_PHYSICAL_EVIDENCE_REUSE_VERIFIED':raise ValueError('Unverified physical reuse')
            return dict(status='GOLDEN_COMMON_BATCH_REGENERATION_PASS',continue_downstream=True,
                physical_evidence_reused=True,new_physics_attempt=False,equivalence=record(equivalent))
    from .conversion_attempt import attempt
    row=attempt(out,sid,'B','TRAIN_pilot',execute=True,resume=True)
    atomic_json(out/'golden_batch/GOLDEN_PHYSICAL_RESULT.json',row)
    if not row.get('task_success') or not row.get('physical_validity'):
        return dict(status='GOLDEN_REPAIRED_PHYSICAL_FULL_TASK_NOT_DEMONSTRATED',continue_downstream=False,row=row)
    from .prototype_media import first_source_full_task
    milestone=first_source_full_task(out,Path(row['attempt']))
    return dict(status='GOLDEN_COMMON_BATCH_REGENERATION_PASS',milestone=milestone,continue_downstream=True)


def pilot(out,resume):
    from .demo_alignment import scored_task_success
    m=read(out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json')
    if record(m['score']['path'])!=m['score'] or not scored_task_success(read(m['score']['path'])):
        raise ValueError('Golden physical prerequisite unmet')
    ids=read(out/'bootstrap/SELECTION.json')['additional_train_source_ids']
    if len(ids)!=5 or len(set(ids))!=5:raise ValueError('Predeclared five-source pilot required')
    path=out/'TRAIN_pilot/PILOT_LEDGER.json';rows=read(path)['rows'] if path.exists() else []
    if rows and not resume:raise FileExistsError(path)
    schedule=[dict(index=2*i+j,source_id=sid,condition=c) for i,sid in enumerate(ids)
              for j,c in enumerate(('A','B') if i%2==0 else ('B','A'))]
    from .conversion_attempt import attempt
    for item in schedule:
        if any(r['index']==item['index'] for r in rows):continue
        print('TRAIN5_START',item['index']+1,10,item['condition'],item['source_id'],flush=True)
        result=plan(out,item['source_id'],item['condition'],'TRAIN_pilot',
            out/'TRAIN_pilot/plan_receipts'/f"{item['index']:02d}_{item['condition']}_{item['source_id']}.json",resume)
        if result.get('full_task_plan'):
            result=attempt(out,item['source_id'],item['condition'],'TRAIN_pilot',execute=True,resume=True)
        else:result=dict(result,terminal=result.get('terminal',result['status']),physical_run=False,task_success=None)
        rows.append(dict(result,**item))
        ledger=dict(status='FIXED_TRAIN5_COMPLETE' if len(rows)==10 else 'FIXED_TRAIN5_RUNNING',
            scheduled_instances=10,completed=len(rows),source_ids=ids,schedule=schedule,rows=rows,
            success_quota_applied=False,episode_specific_repair=False)
        atomic_json(path,ledger)
        text=f"# Current status\n\nRepaired golden common-batch source-conditioned full task verified. Receiving sensitivity and cache regressions passed.\n\nFixed TRAIN5 completed {len(rows)}/10 method-instance attempts. "
        text+='; '.join(f"{c}: {sum(bool(r.get('full_task_plan')) for r in rows if r['condition']==c)} plans, {sum(bool(r.get('physical_run')) for r in rows if r['condition']==c)} physical runs, {sum(r.get('task_success') is True for r in rows if r['condition']==c)} full successes" for c in ('A','B'))
        text+='\n\nCounts describe the completed pilot attempts only. TRAIN40 generation and ACT remain pending pilot review and protocol freeze.\n'
        atomic_text(out/'CURRENT_STATUS.md',text);atomic_text(out/'CHATGPT_UPDATE.md',text)
        with (out/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),stage='TRAIN_pilot',completed=len(rows),scheduled=10,**item,terminal=result['terminal']))+'\n')
        print('TRAIN5_DONE',len(rows),10,result['terminal'],flush=True)
    return read(path)


def dispatch(out,stage,resume):
    from .generalization_gate import require
    require(out, stage)
    if stage=='target_repair':
        result=read(out/'target_repair/RIGHT_RECEIVING_INTEGRATION_RESULT.json')
        return dict(status='RECEIVING_AND_CACHE_REGRESSIONS_VERIFIED',
                    receiving=result,cache=read(out/'target_repair/CACHE_BOUNDARY_REGRESSION.json'))
    if stage=='prototype':return prototype(out,resume)
    if stage=='TRAIN_pilot':return pilot(out,resume)
    if stage=='generation_freeze':
        from .source_contract import run as prepare_source
        prepare_source(out,'TRAIN40')
        from .generation_study import freeze
        return freeze(out)
    from .current_progress import dispatch as existing
    return existing(out,stage,resume)
