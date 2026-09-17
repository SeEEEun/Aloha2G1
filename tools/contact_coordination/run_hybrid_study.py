#!/usr/bin/env python3
"""Thin resumable local entry point; prototype gates prevent invalid evaluation."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_coordination.io import atomic_json,atomic_text,read,record,fingerprint

STAGES=['bootstrap','common_control','source_phase','prototype','pilot','freeze','primary_dev35','paired_ablation','analysis','report','render']
CONTINUATION_STAGES=['inspect','failure_localization','common_control','target_repair','prototype','pilot','freeze','main_dev35','paired_ablation','analysis','paper_report','measured_replays','final_verification']


def signature(out,stage):
    paths=list((ROOT/'tools/contact_coordination').glob('*.py'))+list((ROOT/'configs/contact_coordination').glob('*.json'))
    for relative in ['bootstrap/SPLITS.json','bootstrap/SELECTION.json','source_phase/RESULT.json','prototype/RESULT.json','common_control/RESULT.json']:
        p=out/relative
        if p.exists() and (stage not in ('bootstrap','source_phase','prototype','common_control') or relative.split('/')[0]!=stage):
            paths.append(p)
    return fingerprint(paths)[0]


def perform(out,stage,resume):
    if (out/'CONTINUATION_STATE.json').exists():
        from tools.contact_coordination.continuation_report import perform_stage
        return perform_stage(out,stage,resume)
    if stage=='bootstrap':
        from tools.contact_coordination.hybrid_bootstrap import run
    elif stage=='common_control':
        from tools.contact_coordination.control_audit import run
    elif stage=='source_phase':
        from tools.contact_coordination.source_phase import run
    elif stage=='prototype':
        if (out/'prototype/RESULT.json').exists() and not resume:
            raise FileExistsError('Prototype evidence already exists; use --resume or a new run directory')
        from tools.contact_coordination.prototype import run
        result=run(out,resume)
        from tools.contact_coordination.contact_region import run as region
        from tools.contact_coordination.coupled_fit import run as coupled
        source_id=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
        for name, module, function, manifest in [('contact_region_v2','contact_region',region,'INVOCATION.json'),
                                                ('coupled_pose_region_v3','coupled_fit',coupled,'CONFIG.json')]:
            folder=out/'prototype'/source_id/name
            if resume and (folder/'RESULT.json').exists():
                saved=read(folder/manifest)
                assert record(ROOT/f'tools/contact_coordination/{module}.py')==saved['implementation'], 'Changed development code requires a new run directory'
                if 'source_phase' in saved:
                    assert record(saved['source_phase']['path'])==saved['source_phase']
                if 'source_target' in saved:
                    assert record(saved['source_target']['path'])==saved['source_target']
            else:function(out,resume)
        return result
    elif stage in ('pilot','freeze','primary_dev35','paired_ablation'):
        prototype=read(out/'prototype/RESULT.json') if (out/'prototype/RESULT.json').exists() else {}
        if prototype.get('status')!='SOURCE_CONDITIONED_FULL_TASK_DEMONSTRATED':
            return dict(status='NOT_ATTEMPTED_UPSTREAM',reason='M2 source-conditioned physical full task not demonstrated; no DEV freeze or rollouts authorized by stage prerequisites')
        raise RuntimeError('Downstream study runner is not qualified; do not bypass prototype, controller and collision parity checks')
    elif stage in ('analysis','report'):
        from tools.contact_coordination.analysis_report import run
    elif stage=='render':
        from tools.contact_coordination.prototype_media import run
    return run(out,resume)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=sorted(set(STAGES+CONTINUATION_STAGES+['all'])),default='inspect')
    parser.add_argument('--until',choices=sorted(set(STAGES+CONTINUATION_STAGES)))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.run_dir.resolve()
    if not out.is_relative_to(ROOT/'outputs/contact_coordination_hybrid'):
        parser.error('run directory must be under outputs/contact_coordination_hybrid')
    out.mkdir(parents=True,exist_ok=True)
    stages=CONTINUATION_STAGES if (out/'CONTINUATION_STATE.json').exists() else STAGES
    aliases={'bootstrap':'inspect','primary_dev35':'main_dev35','report':'paper_report','render':'measured_replays'} if stages==CONTINUATION_STAGES else {'inspect':'bootstrap'}
    stage=aliases.get(args.stage,args.stage);until=aliases.get(args.until,args.until)
    if stage!='all' and stage not in stages:parser.error('Stage is not part of this run workflow')
    if until is not None and until not in stages:parser.error('--until stage is not part of this run workflow')
    selected=stages if stage=='all' else stages[stages.index(stage):stages.index(until or stage)+1]
    with (out/'RUN.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another hybrid stage holds this run lock')
        for stage in selected:
            key=signature(out,stage);path=out/'stages'/f'{stage}.json'
            if args.resume and path.exists():
                old=read(path)
                if old.get('signature')==key and old.get('status')!='INFRASTRUCTURE_INVALID' and all(Path(x['path']).is_file() and record(x['path'])==x for x in old.get('artifacts',[])):
                    print('REUSED',stage,old['status'],flush=True);continue
            started=time.monotonic()
            try:result=perform(out,stage,args.resume)
            except Exception as error:
                result=dict(status='SOURCE_EVIDENCE_MISSING' if 'SOURCE_EVIDENCE_MISSING:' in str(error) else 'INFRASTRUCTURE_INVALID',
                            error=repr(error),traceback=traceback.format_exc())
                atomic_json(path,dict(stage=stage,signature=key,**result));raise
            # Capture the stage products; an altered/missing product prevents reuse.
            folders={'bootstrap':['bootstrap'],'inspect':['bootstrap'],'source_phase':['source_phase'],'prototype':['prototype'],
                     'failure_localization':['failure_localization'],'target_repair':['target_repair'],
                     'common_control':['common_control'],'render':['figures','replays'],'measured_replays':['figures','replays']}
            if stage in folders:
                products=[p for directory in folders[stage] for p in (out/directory).rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.csv','.png','.svg','.mp4','.md') and p.name!='MEASURED_CHECKPOINT.npz']
            else:products=[p for p in out.glob('*') if p.is_file() and p.suffix in ('.md','.csv','.json')]
            atomic_json(path,dict(stage=stage,signature=signature(out,stage),runtime_s=time.monotonic()-started,
                                 artifacts=[record(p) for p in sorted(products)],**result))
            log=out/'RUN_LOG.jsonl'
            item=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,status=result['status'],signature=key)
            atomic_text(log,(log.read_text() if log.exists() else '')+json.dumps(item)+'\n')
            print('STAGE',stage,result['status'],flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
