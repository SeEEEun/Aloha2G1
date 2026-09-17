"""Resumable paired ACT training using the authoritative unchanged schedule."""
import copy,os,subprocess,sys,time
from .common import *
from tools import final_paper_act_training_coordinator as legacy
ACTPY=legacy.ACTPY
legacy.AREA=RUN/'act/training'

def main():
    while not (RUN/'act/BRANCH_STATUS.json').exists():time.sleep(15)
    branch=read(RUN/'act/BRANCH_STATUS.json')
    if not branch['paired_ids']:print('ACT_UNAVAILABLE_EMPTY_PAIRED_SET',flush=True);return
    for mode in ('WRIST','INTERACTION','PARITY'):
        expected=RUN/('act/DATASET_PARITY.json' if mode=='PARITY' else f'act/{mode}_DATASET_MANIFEST.json')
        for retry in range(3):
            if expected.exists():break
            legacy.execute([ACTPY,'-m','tools.reconciled_ab.build_dataset',mode],RUN/'act/dataset_logs')
        assert expected.exists(),'Dataset infrastructure needs repair; not an experimental failure'
    assert read(RUN/'act/DATASET_PARITY.json')['status']=='PASS'
    # Primary physical experiment and measured replay rendering finish first,
    # so ACT cannot monopolize the GPU or delay primary result completion.
    while not (RUN/'REFERENCE_MEDIA_COMPLETE.json').exists():time.sleep(15)
    template=read(ROOT/'outputs/paper_core_ab/act_a40/config/train_config.json');configs={}
    for mode in ('WRIST','INTERACTION'):
        dataset=read(RUN/f'act/{mode}_DATASET_MANIFEST.json');cfg=copy.deepcopy(template);cfg['dataset']['root']=dataset['root'];cfg['dataset']['repo_id']=dataset['repo_id'];cfg['output_dir']=str(RUN/'act/training'/mode/'train');cfg['job_name']='reconciled_'+mode.lower();configs[mode]=cfg
    left=copy.deepcopy(configs['WRIST']);right=copy.deepcopy(configs['INTERACTION'])
    for x in (left,right):
        for k in ('output_dir','job_name'):x.pop(k)
        for k in ('root','repo_id'):x['dataset'].pop(k)
    assert left==right
    save(RUN/'act/TRAINING_LAUNCH.json',dict(configs=configs,common_protocol_equal=True,selection=record(RUN/'act/ACT_TRAINING_AND_SELECTION_CONTRACT.json'),dataset_parity=record(RUN/'act/DATASET_PARITY.json'),implementation=record(Path(__file__))))
    environment=subprocess.check_output([ACTPY,'-m','pip','freeze'],text=True,env=legacy.ENV);text(RUN/'act/TRAINING_ENVIRONMENT.txt',environment)
    selected={}
    for mode in ('WRIST','INTERACTION'):
        selected[mode]=legacy.train_mode(mode,configs[mode]);save(RUN/'act/TRAINING_PROGRESS.json',dict(selected=selected,common_config_equal=True))
    save(RUN/'act/TRAINING_COMPLETE.json',dict(selected=selected,selection='Same fixed final100000step checkpoint, predeclared, no DEV outcomes used',inference_mode='RECORDED_SOURCE_IMAGE_CONDITIONED_OPEN_LOOP_WITH_COMMON_SOURCE_CLOCK_DEX3'))
    print('PAIRED_ACT_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
