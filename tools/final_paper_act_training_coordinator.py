#!/usr/bin/env python3
"""Conditional, resumable paired ACT training after primary reference results."""
from pathlib import Path
import copy,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_io import *
ACTPY='/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python'
TRAIN='/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train'
AREA=DEST/'act_training'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')

def execute(args,logdir):
    logdir.mkdir(parents=True,exist_ok=True);path=logdir/f'LOG_{time.time_ns()}.txt'
    with path.open('w') as log:process=subprocess.Popen(args,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(path.with_suffix('.json'),dict(timestamp=now(),pid=process.pid,command=args,status='RUNNING'))
    code=process.wait();atomic_json(path.with_suffix('.json'),dict(timestamp=now(),pid=process.pid,command=args,status='COMPLETE' if code==0 else 'INFRASTRUCTURE_ERROR',returncode=code,log=file_record(path)))
    return code

def dataset_stage(mode):
    expected=DEST/'act_datasets'/('A_B_CORRECTED_DATASET_PARITY_AUDIT.json' if mode=='PARITY' else f'{mode}_DATASET_MANIFEST.json')
    for retry in range(3):
        if expected.exists():return
        execute([ACTPY,str(ROOT/'tools/final_paper_build_act_datasets.py'),mode],AREA/'dataset_logs')
    raise RuntimeError('Dataset infrastructure needs common repair: '+mode)

def train_mode(mode,config):
    final=Path(config['output_dir'])/'checkpoints/100000/pretrained_model';cp=AREA/mode/'train_config.json'
    if not cp.exists():atomic_json(cp,config)
    for retry in range(3):
        if (final/'model.safetensors').exists() and (final.parent/'training_state/training_step.json').exists():
            assert read(final.parent/'training_state/training_step.json')['step']==100000
            result=dict(mode=mode,selected_checkpoint=str(final),selection='Fixed final step100000',files=[file_record(p) for p in sorted(final.rglob('*')) if p.is_file()],config=file_record(cp))
            atomic_json(AREA/mode/'SELECTED_CHECKPOINT.json',result);return result
        checkpoints=sorted(Path(config['output_dir']).glob('checkpoints/[0-9]*/pretrained_model/train_config.json'))
        valid=[p for p in checkpoints if (p.parent/'model.safetensors').exists() and (p.parent.parent/'training_state/optimizer_state.safetensors').exists()]
        if valid:args=[TRAIN,'--config_path',str(valid[-1]),'--resume=true']
        else:
            output=Path(config['output_dir'])
            if output.exists():
                # Preserve a failed initialization that has no resumable
                # checkpoint, then restart the same fixed seed/configuration.
                archive=output.with_name(output.name+f'_incomplete_provenance_{time.time_ns()}');output.rename(archive)
            args=[TRAIN,'--config_path',str(cp)]
        execute(args,AREA/mode/'logs')
    raise RuntimeError('ACT infrastructure requires repair/resume; no scientific unavailability inferred')

def main():
    audit=DEST/'action_dataset_audit/EXACT_ACTION_DATASET_AUDIT.json'
    while not audit.exists():time.sleep(15)
    result=read(audit)
    if result['paired_train_episodes']==0:
        print('ACT_NOT_TRAINABLE_EMPTY_PAIRED_SET',flush=True);return
    while not (DEST/'REFERENCE_PHYSICS_COMPLETE.json').exists():time.sleep(15)
    assert read(DEST/'REFERENCE_PHYSICS_COMPLETE.json')['all_infrastructure_valid']
    for mode in ('WRIST','INTERACTION','PARITY'):dataset_stage(mode)
    assert read(DEST/'act_datasets/A_B_CORRECTED_DATASET_PARITY_AUDIT.json')['unintended_dataset_confounds']==0
    template=read(ROOT/'outputs/paper_core_ab/act_a40/config/train_config.json');configs={}
    for mode in ('WRIST','INTERACTION'):
        cfg=copy.deepcopy(template);dataset=read(DEST/'act_datasets'/f'{mode}_DATASET_MANIFEST.json')
        cfg['dataset']['root']=dataset['root'];cfg['dataset']['repo_id']=dataset['repo_id'];cfg['output_dir']=str(AREA/mode/'train');cfg['job_name']='final_paired_act_'+mode.lower();configs[mode]=cfg
    normalized=[]
    for cfg in configs.values():
        c=copy.deepcopy(cfg)
        for k in ('output_dir','job_name'):c.pop(k)
        for k in ('root','repo_id'):c['dataset'].pop(k)
        normalized.append(c)
    assert normalized[0]==normalized[1]
    atomic_json(AREA/'FINAL_TRAINING_LAUNCH_CONTRACT.json',dict(created_at=now(),configs=configs,common_protocol_equal=True,selection='Fixed final100000-step checkpoint, seed1000, no DEV35 selection',paired_manifest=file_record(DEST/'action_dataset_audit/PAIRED_TRAIN_SET_MANIFEST.json'),dataset_parity=file_record(DEST/'act_datasets/A_B_CORRECTED_DATASET_PARITY_AUDIT.json'),runner=file_record(Path(__file__)),builder=file_record(ROOT/'tools/final_paper_build_act_datasets.py')))
    env=subprocess.check_output([ACTPY,'-m','pip','freeze'],text=True,env=ENV);atomic_text(AREA/'ENVIRONMENT.txt',env)
    selected={mode:train_mode(mode,configs[mode]) for mode in ('WRIST','INTERACTION')}
    atomic_json(AREA/'ACT_TRAINING_COMPLETE.json',dict(selected=selected,common_protocol=True,next_stage='POLICY_SANITY_AND_MATCHED_DEV35_PHYSICS'))
    print('BOTH_PAIRED_ACT_TRAININGS_COMPLETE',flush=True)

if __name__=='__main__':main()
