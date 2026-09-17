"""One bounded source/recipe/method plan, using the existing converter."""
import argparse
import os
import signal
import sys
import time
import cProfile
import pstats
from pathlib import Path
from .io import read,record,atomic_json
from .abc_contract import require,METHODS


def run(out,sid,method,receipt):
    require(out,'TRAIN_calibration')
    if sid not in read(out/'FOLD_FIT_CONTRACT.json')['held_out_source_ids']:
        raise ValueError('Source is outside the predeclared validation fold')
    start=time.monotonic();reads=set();profile=cProfile.Profile();profile.enable()
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).absolute()
            mode=args[1];writing=isinstance(mode,str) and any(k in mode for k in 'wax+')
            if not writing and p.suffix in ('.json','.npz','.parquet','.xml'):
                reads.add(str(p))
                if '/prototype/' in str(p) and not p.is_relative_to(out):
                    raise ValueError('A/B/C worker refuses a prior-run phase solution: '+str(p))
    sys.addaudithook(audit)
    def save(result):
        profile.disable();stats=pstats.Stats(profile)
        timings=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_s=v[2],cumulative_s=v[3])
                 for k,v in stats.stats.items()]
        atomic_json(receipt.with_name('PLANNER_PROFILE.json'),dict(wall_s=time.monotonic()-start,
            top_calls=sorted(timings,key=lambda r:r['cumulative_s'],reverse=True)[:100],
            profiling_changes_search_rules=False,duplicate_evaluation_counts='Not inferred from call count alone'))
        result.update(method_key=method,source_id=sid,worker_wall_s=time.monotonic()-start,
            planning_cap_s=900,prior_world_path_or_q_loaded=False,
            parameters=record(out/'CALIBRATION_PARAMETERS.json'),fold_fit=record(out/'FOLD_FIT_CONTRACT.json'))
        atomic_json(receipt,result);atomic_json(receipt.with_name(receipt.stem+'_READS.json'),sorted(reads));return result
    def timeout(signum,frame):
        save(dict(terminal='NO_PLAN_WITHIN_FIXED_BUDGET',full_task_plan=False,physical_run=False,
                  first_failure='TOTAL_900_SECOND_PLANNING_CAP'))
        raise SystemExit(0)
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    from .source_contract import run as source
    source(out,'TRAIN40',[sid])
    from .conversion_attempt import attempt
    result=attempt(out,sid,METHODS[method]['legacy_key'],'TRAIN_calibration',execute=False,resume=True)
    signal.alarm(0);return save(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--source-id',required=True)
    p.add_argument('--method',choices=list(METHODS),required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();run(a.run_dir.resolve(),a.source_id,a.method,a.receipt.resolve())
