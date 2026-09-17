#!/usr/bin/env python3
"""Resumable local study. No hardware or policy training entry points."""
from pathlib import Path
import argparse, importlib, os, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_coordination.io import DEFAULT_RUN, atomic_json, atomic_text, read, record

STAGES=['bootstrap','common_control','source_graph','prototype','shared_smoke','protocol_freeze','dev35','layout_tests','analysis','render','report']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--stage',choices=STAGES,default='bootstrap')
    p.add_argument('--until',choices=STAGES,default='report')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (run/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(os.getpid()));lock.flush()
        start=time.monotonic()
        for stage in STAGES[STAGES.index(a.stage):STAGES.index(a.until)+1]:
            print('STAGE_START',stage,flush=True)
            status=run/'stages'/f'{stage}.json'
            try:
                module=importlib.import_module('tools.contact_coordination.'+stage)
                result=module.run(run,resume=a.resume)
                atomic_json(status,dict(stage=stage,elapsed_sec=time.monotonic()-start,**result))
                print('STAGE_END',stage,result.get('status'),flush=True)
            except Exception as e:
                atomic_json(status,dict(stage=stage,status='INFRASTRUCTURE_INVALID',error=repr(e),traceback=traceback.format_exc()))
                raise
            atomic_text(run/'CHATGPT_UPDATE.md',f'# Local study checkpoint\n\nLast completed stage: {stage}. Status: {result.get("status")}.\n\nResume with the documented local entry point. No automatic ChatGPT notifications are configured.\n')
    return 0

if __name__=='__main__': raise SystemExit(main())
