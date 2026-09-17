#!/usr/bin/env python3
"""Common component tests, independent of frozen representation outcomes."""
from pathlib import Path
import json,os,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_run import OUT,DEST,INITIAL,read,file_record,atomic_json,atomic_text,now
ISAAC='/home/jbnu/miniconda3/envs/isaaclab6/bin/python'
OLD=ROOT/'outputs/final_episode_registered_eval35/00_qualification'
D=DEST/'dex3'

def call(command,folder,required):
    if all((folder/p).exists() for p in required):return
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'INVOCATION.json',dict(command=command))
    for retry in range(3):
        with (folder/f'ENGINE_{retry}_{time.time_ns()}.log').open('w') as log:
            r=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        if all((folder/p).exists() for p in required):return
        print('DEX3_INFRA_RETRY',folder.name,retry,r.returncode,flush=True)
    raise RuntimeError(f'Common component infrastructure retry exhausted: {folder}')

def main():
    # The zero-contact process was launched before this coordinator. Do not
    # duplicate it. Read only its final atomic result when it becomes available.
    zero=D/'zero_contact/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json'
    while not zero.exists():time.sleep(10)
    print('DEX3_ZERO',read(zero)['status'],flush=True)
    loaded=D/'gravity_loaded'
    call([ISAAC,str(ROOT/'tools/final_paper_loaded_dex3_audit.py'),'--arm-state-json',str(INITIAL),'--headless','--solver-position-iterations','80','--solver-velocity-iterations','4','--output-dir',str(loaded)],loaded,['DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json'])
    old=read(OLD/'PROVISIONAL_QUALIFICATION_FREEZE.json')
    freeze=dict(old);freeze['files']=[file_record(Path(r['path'])) for r in old['files']]
    freeze['created_at']=now();freeze['prior_manifest']=file_record(OLD/'PROVISIONAL_QUALIFICATION_FREEZE.json')
    freeze['purpose']='Fresh common loaded-component qualification; latest shared runtime bytes, independent of A/B cohort outcomes'
    fp=D/'COMMON_COMPONENT_PROVISIONAL_FREEZE.json';atomic_json(fp,freeze)
    tests=[]
    for side in ('left','right','scripted'):
        folder=D/f'contact_{side}';full=side=='scripted'
        command=OLD/'contact_seeking_commands'/('scripted_full_task_contact_seeking.npz' if full else f'{side}_standalone_contact_seeking.npz')
        invocation=[ISAAC,str(ROOT/'tools/run_direct_physical_execution_isaac.py'),'--qualification-mode','--direct-freeze-manifest',str(fp),'--config',str(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json'),'--side','right' if full else side,'--geometry','INTERMEDIATE_PLUSH_PROXY','--profile','P14','--output-dir',str(folder),'--scripted-command-path',str(command),'--object-spawn-side','left' if full else side,'--headless']
        if full:invocation+=['--object-registration-config',str(OLD/'SCRIPTED_QUALIFICATION_OBJECT_REGISTRATION.json'),'--audit-robot-bin','--full-task-audit','--bin-height-m','0.150','--bin-rim-bevel-m','0.003']
        call(invocation,folder,['event_log.npz','trial_result.json','DIRECT_EXECUTION_RUNTIME_SUMMARY.json'])
        specs=read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs'];lo=np.array([r['minimum'] for r in specs]);hi=np.array([r['maximum'] for r in specs])
        with np.load(folder/'event_log.npz') as z:
            measured=z['MEASURED_Q'];commanded=z['EXECUTED_COMMAND']
            contacts={s:{d:float(z[f'{s}_{d}_force_n'].max()) for d in ('thumb','index','middle')} for s in ('left','right')}
            mv=np.sum((measured[:,14:]<lo[14:]-1e-6)|(measured[:,14:]>hi[14:]+1e-6),axis=0)
            cv=np.sum((commanded[:,14:]<lo[14:])|(commanded[:,14:]>hi[14:]),axis=0)
            finite=bool(np.isfinite(measured).all() and np.isfinite(commanded).all())
        report=dict(test=side,measured_violations_per_joint=mv.tolist(),commanded_violations_per_joint=cv.tolist(),finite=finite,contacts=contacts,
            pass_limits=bool(not mv.any() and not cv.any() and finite),trace=file_record(folder/'event_log.npz'),runtime=read(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json'),trial=read(folder/'trial_result.json'))
        atomic_json(folder/'ALL14_LOADED_LIMITS.json',report);tests.append(report)
        print('DEX3_LOADED',side,report['pass_limits'],flush=True)
    audits=[read(zero),read(loaded/'DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json')]
    passed=all(r['status']=='PASS' for r in audits) and all(r['pass_limits'] for r in tests)
    result=dict(status='PASS' if passed else 'COMMON_DEX3_COMPONENT_NOT_QUALIFIED',zero_and_gravity_audits=[file_record(zero),file_record(loaded/'DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json')],loaded_tests=tests,
        common_all14_mapping_sign_readback_runtime_limit_pass=passed,measured_state_clipping_used=False,qualification_independent_of_representation=True)
    atomic_json(D/'COMMON_DEX3_QUALIFICATION.json',result)
    if passed:
        final=dict(freeze,status='FROZEN_BEFORE_EVAL35',qualification=file_record(D/'COMMON_DEX3_QUALIFICATION.json'))
        atomic_json(OUT/'03_common_execution_freeze/COMMON_DEX3_FREEZE_MANIFEST.json',final)
    else:
        atomic_text(D/'COMPONENT_DIAGNOSTIC.md','# Common Dex3 qualification requires infrastructure diagnosis\n\n'+json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
