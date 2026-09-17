"""Hash-compatible component reuse plus a fresh common measured control."""
import subprocess,sys,os,time
import numpy as np
from .common import *
from tools.final_paper_reference_physics import ISAAC,PHYSICS_CONFIG,JOINT
from tools.direct_physical_execution_layer import authoritative_joint_limits

def main():
    prior=read(PRIOR/'dex3/COMMON_COMPONENT_PROVISIONAL_FREEZE.json')
    for dep in prior['files']:assert record(dep['path'])==dep
    assert read(PRIOR/'dex3/COMMON_DEX3_QUALIFICATION.json')['status']=='PASS'
    assert read(PRIOR/'dex3/SOURCE_CLOCK_QUALIFICATION.json')['status']=='PASS'
    command=PRIOR/'dex3/SOURCE_CLOCK_FAST_TRAIN_COMPONENT_COMMAND.npz'
    paths={Path(r['path']) for r in prior['files']};paths.update([ROOT/'tools/final_paper_source_clock_dex3.py',ROOT/'tools/final_paper_physics_isaac.py',ROOT/'tools/reconciled_ab/physics_engine.py',ROOT/'tools/reconciled_ab/runtime_geometry.py',command])
    freeze=dict(status='PRE_EVAL35_QUALIFICATION_PROVISIONAL',graspability_classifier_used=False,files=[record(p) for p in sorted(paths)],purpose='Common non-test component control; no A/B trajectory selection or tuning')
    fp=RUN/'components/COMPONENT_FREEZE.json';save(fp,freeze)
    folder=RUN/'components/known_control';folder.mkdir(parents=True,exist_ok=True)
    invocation=[ISAAC,str(ROOT/'tools/reconciled_ab/physics_engine.py'),'--qualification-mode','--direct-freeze-manifest',str(fp),'--config',str(PHYSICS_CONFIG),'--side','right','--geometry','INTERMEDIATE_PLUSH_PROXY','--profile','P14','--output-dir',str(folder),'--scripted-command-path',str(command),'--object-spawn-side','left','--object-registration-config',str(ROOT/'outputs/final_episode_registered_eval35/00_qualification/SCRIPTED_QUALIFICATION_OBJECT_REGISTRATION.json'),'--audit-robot-bin','--full-task-audit','--bin-height-m','0.150','--bin-rim-bevel-m','0.003','--headless']
    save(folder/'INVOCATION.json',dict(command=invocation,component_only=True,source_command=record(command)))
    for retry in range(3):
        if all((folder/x).exists() for x in ('event_log.npz','trial_result.json','DIRECT_EXECUTION_RUNTIME_SUMMARY.json')):break
        with (folder/f'ENGINE_{retry}.log').open('w') as f:r=subprocess.run(invocation,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
        print('COMMON_CONTROL_ENGINE',retry,r.returncode,flush=True)
    with np.load(folder/'event_log.npz') as z:
        lo,hi,names=authoritative_joint_limits(read(JOINT));q=z['MEASURED_Q'];cmd=z['EXECUTED_COMMAND'];frames=z['control_frame'].astype(int)
        with np.load(command) as source:complete=np.array_equal(np.unique(frames),np.arange(len(source['commanded_q_rad'])))
        measured=np.count_nonzero((q[:,14:]<lo[14:]-1e-6)|(q[:,14:]>hi[14:]+1e-6),axis=0);commanded=np.count_nonzero((cmd[:,14:]<lo[14:])|(cmd[:,14:]>hi[14:]),axis=0)
    valid=complete and np.isfinite(q).all() and not measured.any() and not commanded.any()
    result=dict(status='PASS' if valid else 'FAIL',complete_trace=bool(complete),measured_violations=measured.tolist(),commanded_violations=commanded.tolist(),mapping_sign_readback_runtime_limits_reused_by_exact_dependency_hash=True,
        old_component_qualification=record(PRIOR/'dex3/COMMON_DEX3_QUALIFICATION.json'),source_clock_qualification=record(PRIOR/'dex3/SOURCE_CLOCK_QUALIFICATION.json'),fresh_trace=record(folder/'event_log.npz'),runtime_collision=record(folder/'RUNTIME_COLLISION_MODEL.json'),component_only=True,dev35_rollouts=0)
    save(RUN/'components/COMPONENT_QUALIFICATION.json',result);print('COMMON_COMPONENT',result['status'],flush=True)
    log('COMMON_RUNTIME_COMPONENT','COMPLETE' if valid else 'NEEDS_COMMON_DIAGNOSIS','Hash-compatible all14 qualification checked and fresh loaded/contact source-clock control executed.',artifacts=[RUN/'components/COMPONENT_QUALIFICATION.json'],next_stage='TRAIN_REFERENCE_PILOT_AND_PHYSICS_FREEZE')

if __name__=='__main__':main()
