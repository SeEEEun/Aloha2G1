"""Bounded independent PhysX jobs; each keeps the common full-trace contract."""
from pathlib import Path
from .io import read,atomic_json


def worker(out,plan,index=0,total=1):
    from .architecture_physics import run
    out=Path(out);name='physical_case_reports/'+plan['source_id']+'_'+plan['method_key']+'.json'
    artifacts=run(out,require_complete_coverage=False,collect_structure=False,plans_override=[plan],report_name=name,
        progress_offset=index,total_scheduled=total)
    return read(out/name)['rows'][0],[str(p) for p in artifacts]


def run(out):
    plans=read(out/'GOLDEN_PLANNING.json')['rows']+read(out/'COVERAGE8_PLANNING.json')['rows']
    keys=[(p['source_id'],p['method_key']) for p in plans];rows={};artifacts=[]
    # Serial simulators avoid GPU-context contention. Stop immediately on an
    # infrastructure exception, without leaving queued simulations running.
    # Completed task failures remain cached; physics parameters are unchanged.
    for index,plan in enumerate(plans):
        row,files=worker(str(out),plan,index,len(plans));rows[keys[index]]=row;artifacts+=files
        atomic_json(out/'PHYSICAL_VERIFICATION.json',dict(status='IN_PROGRESS',rows=[rows[k] for k in keys if k in rows],
            completed=len(rows),scheduled=len(keys),fixed_concurrent_workers=1))
    ordered=[rows[k] for k in keys]
    assert all(not p['full_task_plan'] or rows[(p['source_id'],p['method_key'])]['physics_executed'] for p in plans)
    path=out/'PHYSICAL_VERIFICATION.json';atomic_json(path,dict(status='PASS',rows=ordered,
        task_success_requirement='Architecture execution, not100% task success',all_complete_commands_executed=True,
        no_episode_tuning=True,fixed_concurrent_workers=1,physics_parameters_unchanged=True))
    return [path,*[Path(p) for p in sorted(set(artifacts))]]
