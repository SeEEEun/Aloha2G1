"""Independent fixed-source planning jobs under the one locked supervisor."""
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from .io import read,atomic_json


def worker(out,sid,method,journal):
    os.environ['SOURCE_GUIDED_RRT_AUDIT_LOG']=str(journal)
    from .interaction_chain import attempt
    return attempt(Path(out),sid,method)


def run(out,golden=False):
    fixed=read(out/'COVERAGE8.json');sources=[fixed['golden']] if golden else fixed['source_ids']
    jobs=[(sid,m) for sid in sources for m in ('B_INDEPENDENT','C_COUPLED')];workers=min(4,len(jobs))
    main_journal=Path(os.environ['SOURCE_GUIDED_RRT_AUDIT_LOG']);rows={};journals=[]
    # The structural suite and completed Golden plans already passed. Exercise
    # that same physical pipeline while the independent Coverage8 plans run.
    # Final physics verification validates/reuses the immutable trial receipts;
    # this preflight never selects candidates or reports Coverage8 performance.
    preflight=None;preflight_error=[]
    if not golden:
        import threading,traceback
        def verify_golden():
            try:
                from .architecture_physics import run as physics
                physics(out,require_complete_coverage=False,collect_structure=False,
                    plans_override=read(out/'GOLDEN_PLANNING.json')['rows'],report_name='GOLDEN_PHYSICAL_PREFLIGHT.json')
            except Exception:
                preflight_error.append(traceback.format_exc())
                atomic_json(out/'GOLDEN_PREFLIGHT_ERROR.json',dict(error=preflight_error[-1],ordinary_repair_required=True))
        preflight=threading.Thread(target=verify_golden,daemon=False);preflight.start()
    # Four independent sources share no scientific output directory. Budget,
    # seeds, selected source set and result order are unchanged. Workers belong
    # to the supervisor's process group, so invalidation terminates all of them.
    path=out/('GOLDEN_PLANNING.json' if golden else 'COVERAGE8_PLANNING.json')
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        pending={}
        for sid,method in jobs:
            journal=main_journal.with_name(main_journal.stem+'_'+sid+'_'+method+'.jsonl');journals.append(journal)
            pending[pool.submit(worker,str(out),sid,method,str(journal))]=(sid,method)
        for future in as_completed(pending):
            key=pending[future];rows[key]=future.result()
            atomic_json(path,dict(status='IN_PROGRESS',rows=[rows[k] for k in jobs if k in rows],
                completed=len(rows),scheduled=len(jobs),fixed_concurrent_workers=workers))
    ordered=[rows[k] for k in jobs]
    atomic_json(path,dict(status='COMPLETE',rows=ordered,complete=sum(r['full_task_plan'] for r in ordered),
        fixed_concurrent_workers=workers,per_connection_budget_unchanged=True))
    if preflight:
        preflight.join()
        if preflight_error:raise RuntimeError('Golden physical preflight failed; see GOLDEN_PREFLIGHT_ERROR.json')
    return [path,*[Path(r['receipt']) for r in ordered],*[p for p in journals if p.exists()]]
