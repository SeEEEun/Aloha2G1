"""Thin locked stage entry point for the existing converter's A/B/C study."""
import argparse
import fcntl
import json
import time
from pathlib import Path
from .io import ROOT, read, record, atomic_json, atomic_text
from .abc_contract import require

STAGES = ['inspect', 'environment_golden_audit', 'core_regression',
          'video_contract_test', 'parameter_inventory', 'folds',
          'bounded_calibration', 'final_refit', 'freeze', 'TRAIN40_ABC',
          'DEV35_ABC_reference', 'ACT_BC', 'ACT_DEV35',
          'full_diagnostics', 'render', 'statistics', 'report', 'status']


def dispatch(out, stage, resume):
    require(out, stage)
    if stage in ('inspect', 'environment_golden_audit', 'parameter_inventory', 'folds', 'report'):
        from .abc_audit import run
        return run(out, stage, resume)
    if stage == 'core_regression':
        from .abc_regression import run
        return run(out, resume)
    if stage == 'video_contract_test':
        from .abc_video_control import run
        return run(out, resume)
    if stage == 'bounded_calibration':
        from .abc_calibration import run
        return run(out, resume)
    if stage == 'status':
        return read(out/'ABC_STUDY.json')
    raise RuntimeError('ABC_STAGE_NOT_IMPLEMENTED_OR_QUALIFIED: '+stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args();out = args.run_dir.resolve()
    out.relative_to(ROOT/'outputs/train40_calibrated_abc')
    with (out/'.abc_study.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = dispatch(out, args.stage, args.resume)
        except RuntimeError as error:
            if not str(error).startswith(('ABC_GATE_CLOSED:', 'ABC_STAGE_NOT_IMPLEMENTED_OR_QUALIFIED:')):
                raise
            result = dict(status='NOT_ATTEMPTED_PREREQUISITE', reason=str(error),
                          continue_downstream=False, return_to_permitted_work=True)
        receipt = dict(stage=args.stage, result=result, time=time.time(), entrypoint=record(__file__))
        atomic_json(out/'stage_receipts'/f'{args.stage}.json', receipt)
        with (out/'RUN_LOG.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt)+'\n');stream.flush()
        print(json.dumps(result), flush=True)
        if result.get('status') == 'NOT_ATTEMPTED_PREREQUISITE':
            return 3
        return 1 if result.get('status') in ('FAIL', 'INFRASTRUCTURE_REPAIR_REQUIRED') else 0


if __name__ == '__main__':
    raise SystemExit(main())
