"""Bounded invocation of the existing per-source batch converter (no physics)."""
import argparse,signal,time,sys,os,json
from pathlib import Path
from .io import ROOT,read,record,atomic_json


def run(out,sid,condition,scope,receipt):
    started=time.monotonic();reads=set()
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).absolute()
            mode=args[1];writing=isinstance(mode,str) and any(c in mode for c in 'wax+')
            if not writing and path.suffix in ('.json','.npz','.parquet','.xml'):
                reads.add(str(path))
                # Previous solutions are provenance, never worker inputs.
                if '/prototype/' in str(path) and not path.is_relative_to(out):
                    raise ValueError('Batch worker refuses a prior-run prototype solution: '+str(path))
    sys.addaudithook(audit)
    def save(result):
        result.update(worker_runtime_s=time.monotonic()-started,planning_budget_s=900,
                      source_id=sid,condition=condition,scope=scope,
                      original_source_reparsed_or_hash_verified=True,saved_golden_trajectory_loaded=False)
        atomic_json(receipt,result)
        atomic_json(receipt.with_name(receipt.stem+'_READS.json'),sorted(reads))
        return result
    def cutoff(signum,frame):
        save(dict(status='NO_PLAN_WITHIN_FIXED_BUDGET',full_task_plan=False,physical_run=False,
                  first_failure='COMMON_BATCH_900_SECOND_LIMIT'))
        raise SystemExit(0)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(900)
    from .source_contract import run as source_contract
    source_contract(out,'DEV35' if scope=='REFERENCE_coupling10' else 'TRAIN40',[sid])
    from .conversion_attempt import attempt
    result=attempt(out,sid,condition,scope,execute=False,resume=True)
    signal.alarm(0)
    result['status']=result['terminal']
    save(result)
    print(result['status'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source-id',required=True);p.add_argument('--condition',choices=['A','B','B_NO_COUPLING'],required=True)
    p.add_argument('--scope',choices=['TRAIN_pilot','TRAIN40_conversion','REFERENCE_coupling10'],default='TRAIN_pilot')
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    run(a.run_dir.resolve(),a.source_id,a.condition,a.scope,a.receipt.resolve())
