"""Thin paired launcher for the installed ACT trainer and its checkpoint resume.

Preparation requires real matched target-domain datasets and a verified full
source task. It does not create supervision or select checkpoints from DEV35.
"""
from pathlib import Path
import copy
import fcntl
import os
import subprocess
import time
from .io import ROOT,read,record,atomic_json

TRAIN='/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train'


def verify_records(records):
    for r in records:
        if record(r['path'])!=r:raise ValueError('Dependency changed: '+r['path'])


def prepare(out,dataset_a,dataset_b,parity_path):
    from .generalization_gate import require
    require(out, 'ACT_train_or_verified_reuse')
    from .demo_alignment import scored_task_success
    milestone=read(out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json')
    verify_records([milestone['score']])
    score=read(milestone['score']['path']);verify_records([score['trace']])
    if not scored_task_success(score):raise ValueError('Full source-conditioned task not verified')
    parity=read(parity_path)
    if parity['status']!='VERIFIED_EFFECTIVE_TARGET_DATASET_PARITY':raise ValueError('Effective dataset audit incomplete')
    paths={'A':Path(dataset_a),'B':Path(dataset_b)}
    if parity['dataset_manifests']!={k:record(v) for k,v in paths.items()}:raise ValueError('Audited dataset manifests differ')
    datasets={k:read(p) for k,p in paths.items()}
    ids={k:[e['source_id'] for e in d['episodes']] for k,d in datasets.items()}
    split=read(out/'SPLIT_CONTRACT.json')
    if split['status']!='AUTHORIZED_TRAIN40_DEV35':raise ValueError('TRAIN40 authorization required')
    if not ids['A'] or ids['A']!=ids['B'] or len(set(ids['A']))!=len(ids['A']):raise ValueError('Expected identical nonempty source membership')
    if not set(ids['A'])<=set(split['authorized_training_source_ids']) or set(ids['A'])&set(split['evaluation_source_ids']):
        raise ValueError('Training membership violates the authorized split')
    protocol_path=out/'act_training/SHARED_PROTOCOL_PREDECLARATION.json';protocol=read(protocol_path)
    verify_records(protocol['implementation_files'])
    area=out/'act_training';configs={}
    for condition,d in datasets.items():
        verify_records(d['files']);verify_records([d['contract']['manifest']])
        membership=read(d['contract']['manifest']['path'])
        if membership['paired_source_ids']!=ids[condition]:raise ValueError('Changed frozen membership')
        info=read(Path(d['root'])/'meta/info.json')
        if info['total_episodes']!=len(ids[condition]) or info['fps']!=30:raise ValueError('Dataset count/rate mismatch')
        cfg=copy.deepcopy(protocol['shared_config'])
        cfg['dataset'].update(root=d['root'],repo_id='local/hybrid_g1_'+condition.lower())
        cfg.update(output_dir=str((area/condition/'train').resolve()),job_name='hybrid_g1_act_'+condition.lower())
        configs[condition]=cfg
    canonical=[]
    for cfg in configs.values():
        c=copy.deepcopy(cfg)
        for key in ('output_dir','job_name'):c.pop(key)
        for key in ('root','repo_id'):c['dataset'].pop(key)
        canonical.append(c)
    if canonical[0]!=canonical[1]:raise ValueError('A/B learning protocols differ')
    contract=dict(status='PREPARED_NOT_TRAINED',paired_source_ids=ids['A'],configs=configs,
        selection='Same fixed final training step; no DEV35 checkpoint selection',
        dependencies=[record(protocol_path),record(parity_path),record(out/'SPLIT_CONTRACT.json'),*[record(p) for p in paths.values()],
                      *protocol['implementation_files'],record(__file__)])
    path=area/'PAIRED_LAUNCH_CONTRACT.json'
    if path.exists() and read(path)!=contract:raise ValueError('Immutable training contract changed')
    for condition,cfg in configs.items():atomic_json(area/condition/'train_config.json',cfg)
    atomic_json(path,contract)
    return contract


def train(out,resume=False):
    from .generalization_gate import require
    require(out, 'ACT_train_or_verified_reuse')
    area=out/'act_training';contract_path=area/'PAIRED_LAUNCH_CONTRACT.json';contract=read(contract_path)
    verify_records(contract['dependencies'])
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    selected={}
    with (area/'.paired_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for condition,cfg in contract['configs'].items():
            checkpoint_root=Path(cfg['output_dir'])/'checkpoints'
            checkpoints=[]
            for p in checkpoint_root.glob('[0-9]*/pretrained_model/train_config.json'):
                state=p.parent.parent/'training_state'
                if (p.parent/'model.safetensors').exists() and (state/'optimizer_state.safetensors').exists() and (state/'training_step.json').exists():
                    checkpoints.append((int(read(state/'training_step.json')['step']),p))
            checkpoints.sort(key=lambda pair:pair[0])
            final=next((p for step,p in checkpoints if step==cfg['steps']),None)
            if final is None:
                if checkpoints:
                    if not resume:raise ValueError('Existing optimizer checkpoint: use --resume')
                    args=[TRAIN,'--config_path',str(checkpoints[-1][1]),'--resume=true']
                else:
                    if Path(cfg['output_dir']).exists():raise ValueError('Incomplete initialization without optimizer checkpoint: diagnose and preserve it before retry')
                    args=[TRAIN,'--config_path',str(area/condition/'train_config.json')]
                log=area/condition/'logs'/f'{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('w') as stream:
                    process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    atomic_json(log.with_suffix('.json'),dict(status='RUNNING',pid=process.pid,command=args,contract=record(contract_path)))
                    code=process.wait()
                atomic_json(log.with_suffix('.json'),dict(status='FINISHED' if code==0 else 'INFRASTRUCTURE_ERROR',returncode=code,command=args,log=record(log),contract=record(contract_path)))
                if code:raise RuntimeError(f'ACT-{condition} training failed; inspect {log} and resume the same protocol after repair')
                # The trainer is authoritative about step formatting.
                finals=[p for p in checkpoint_root.glob('[0-9]*/pretrained_model/train_config.json')
                        if read(p.parent.parent/'training_state/training_step.json')['step']==cfg['steps']]
                if len(finals)!=1:raise ValueError('Expected one selected final-step checkpoint')
                final=finals[0]
            selected[condition]=dict(checkpoint=str(final.parent),step=cfg['steps'],
                files=[record(p) for p in sorted(final.parent.rglob('*')) if p.is_file()],
                lineage=record(contract_path),policy_interface_validation='STILL_REQUIRED')
            atomic_json(area/condition/'SELECTED_CHECKPOINT.json',selected[condition])
    result=dict(status='BOTH_FIXED_STEP_TRAININGS_RECORDED_INTERFACE_TEST_REQUIRED',selected=selected)
    atomic_json(area/'ACT_TRAINING_COMPLETE.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--dataset-a',type=Path);p.add_argument('--dataset-b',type=Path);p.add_argument('--parity',type=Path);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    print(prepare(a.run_dir.resolve(),a.dataset_a,a.dataset_b,a.parity) if a.prepare else train(a.run_dir.resolve(),a.resume))
