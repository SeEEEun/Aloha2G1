#!/usr/bin/env python3
"""TRAIN-calibrated fast-close component stress plus method-blind timing tests."""
from pathlib import Path
import json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_reference_physics import *
from tools.final_paper_source_clock_dex3 import SourceClockDex3
from tools.common_execution_layer import ExecutionSnapshot
from tools.final_paper_dex3_qualification import call,OLD,D

def main():
    p=D/'SOURCE_CLOCK_QUALIFICATION.json'
    if p.exists():return
    while not (D/'COMMON_DEX3_QUALIFICATION.json').exists():time.sleep(10)
    assert read(D/'COMMON_DEX3_QUALIFICATION.json')['status']=='PASS'
    primitive=Dex3Primitive.from_frozen_dependencies(read(PHYSICS_CONFIG),read(PHYSICAL_ENVIRONMENT),read(COMMON_PHYSICAL_CONTROLLER))
    lo,hi,names=authoritative_joint_limits(read(JOINT));natural=np.array(read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad'])
    minimum_close=[];offline=[]
    for row in cases():
        if row['group']!='TRAIN40':continue
        with np.load(row['source']['path']) as z:hand,intent,e,_=nominal_from_source(z)
        minimum_close.extend([e['LEFT_CLOSE_COMPLETE']-e['LEFT_CLOSE_BEGIN'],e['RIGHT_ACQUIRE_SOURCE']-e['RIGHT_CLOSE_BEGIN']])
        safe=np.hstack((np.tile(natural,(len(hand),1)),hand));ctrl=SourceClockDex3(primitive,intent,safe,safe,lo,hi,e,hand)
        measured=safe[0];states=[];commands=[]
        for f in range(len(safe)):
            snapshot=ExecutionSnapshot(measured,np.eye(4),{s:np.eye(4) for s in ('left','right')},{s:{d:0. for d in ('thumb','index','middle')} for s in ('left','right')},1.)
            decision=ctrl.step(f,snapshot);measured=decision.executed_command;commands.append(measured);states.append(dict(ctrl.grasp_state))
        q=np.array(commands)
        assert np.array_equal(q[:,:14],safe[:,:14])
        assert np.all(q>=lo) and np.all(q<=hi)
        assert np.allclose(q[:21,14:],hand[:21],atol=1e-12,rtol=0)
        assert np.allclose(q[e['LEFT_CLOSE_COMPLETE'],14:21],ctrl.left_full_close,atol=1e-12)
        assert states[e['LEFT_CLOSE_BEGIN']]['left']=='PROGRESSIVE_CLOSE'
        assert states[e['RIGHT_CLOSE_BEGIN']]['right']=='PROGRESSIVE_CLOSE'
        assert all(v is None for v in ctrl.grasp_confirmed_frame.values())
        offline.append(dict(key=row['key'],status='PASS',source_clock=e))
    atomic_json(D/'SOURCE_CLOCK_OFFLINE_TRAIN40.json',dict(results=offline,minimum_train_close_frames=min(minimum_close),no_contact_cannot_confirm=True,targets_or_events_changed=False))
    # Component-only synthetic timing stress, not an experimental source
    # trajectory: same prequalified loaded arm program, fastest TRAIN close.
    with np.load(OLD/'contact_seeking_commands/scripted_full_task_contact_seeking.npz') as z:archive={k:z[k].copy() for k in z.files}
    original=archive['commanded_q_rad'];intent=archive['common_task_intent'].astype(str);count=len(intent)
    start=lambda name:int(np.flatnonzero(intent==name)[0])
    lc=start('LEFT_CLOSE_INTENT');ra=start('HANDOFF_INTENT');release=start('RIGHT_HOLD_INTENT');final=start('FINAL_RELEASE_INTENT');fast=min(minimum_close)
    events={'APPROACH_START':lc,'LEFT_CLOSE_BEGIN':lc+primitive.preshape_frames,'LEFT_CLOSE_COMPLETE':lc+primitive.preshape_frames+fast,
        'LEFT_GRASP_SOURCE':lc+primitive.preshape_frames+fast,'LEFT_LIFT_BEGIN':start('LEFT_HOLD_INTENT'),
        'RIGHT_APPROACH_BEGIN':ra,'RIGHT_CLOSE_BEGIN':ra+primitive.preshape_frames,'RIGHT_ACQUIRE_SOURCE':ra+primitive.preshape_frames+fast,
        'LEFT_RELEASE_BEGIN':release,'RIGHT_OWNERSHIP_SOURCE':release,'FINAL_RELEASE_BEGIN':final,'TASK_END':count-1}
    hand=np.empty((count,14))
    for side,offset in [('left',0),('right',7)]:
        op=getattr(primitive,f'{side}_open');pre=getattr(primitive,f'{side}_preshape');closed=getattr(primitive,f'{side}_full_close')
        begin=events['APPROACH_START'] if side=='left' else events['RIGHT_APPROACH_BEGIN'];close=events['LEFT_CLOSE_BEGIN'] if side=='left' else events['RIGHT_CLOSE_BEGIN'];end=events['LEFT_CLOSE_COMPLETE'] if side=='left' else events['RIGHT_ACQUIRE_SOURCE'];rel=release if side=='left' else final
        for f in range(count):
            if f<begin:v=op
            elif f<close:v=op+(pre-op)*(f-begin)/max(1,close-begin)
            elif f<end:v=pre+(closed-pre)*(f-close)/max(1,end-close)
            elif f<rel:v=closed
            else:v=closed+(op-closed)*min(1,(f-rel)/primitive.release_frames)
            hand[f,offset:offset+7]=v
    hand=np.clip(hand,lo[14:]+.005,hi[14:]-.005);values=np.hstack((original[:,:14],hand))
    archive.update(commanded_q_rad=values,raw_policy_command=values,common_initial_q_rad=values[0],common_source_nominal_dex3=hand,source_event_names=np.asarray(list(events)),execution_event_frames=np.asarray(list(events.values())))
    cmd=D/'SOURCE_CLOCK_FAST_TRAIN_COMPONENT_COMMAND.npz';atomic_npz(cmd,**archive)
    provisional=read(D/'COMMON_COMPONENT_PROVISIONAL_FREEZE.json');provisional['files'] += [file_record(ROOT/'tools'/name) for name in ('final_paper_source_clock_dex3.py','final_paper_physics_isaac.py','final_paper_source_clock_qualification.py')]
    provisional['files'].append(file_record(cmd));fp=D/'SOURCE_CLOCK_COMPONENT_FREEZE.json';atomic_json(fp,provisional)
    folder=D/'source_clock_loaded_stress'
    invocation=[ISAAC,str(ROOT/'tools/final_paper_physics_isaac.py'),'--qualification-mode','--direct-freeze-manifest',str(fp),'--config',str(PHYSICS_CONFIG),'--side','right','--geometry','INTERMEDIATE_PLUSH_PROXY','--profile','P14','--output-dir',str(folder),'--scripted-command-path',str(cmd),'--object-spawn-side','left','--object-registration-config',str(OLD/'SCRIPTED_QUALIFICATION_OBJECT_REGISTRATION.json'),'--audit-robot-bin','--full-task-audit','--bin-height-m','0.150','--bin-rim-bevel-m','0.003','--headless']
    # Component acceptance is derived directly from the complete saved physical
    # trace. A post-simulation JSON-summary error does not erase valid physics.
    # Preserve the executed-code freeze separately when reusing such a trace.
    call(invocation,folder,['event_log.npz'])
    with np.load(folder/'event_log.npz') as z:
        frames=z['control_frame'].astype(int)
        assert np.array_equal(np.unique(frames),np.arange(count)), 'Incomplete stress trace'
        assert np.array_equal(z['RAW_POLICY_COMMAND'],values[frames]), 'Stress trace/command mismatch'
        measured=z['MEASURED_Q'];executed=z['EXECUTED_COMMAND'];mv=np.sum((measured[:,14:]<lo[14:]-1e-6)|(measured[:,14:]>hi[14:]+1e-6),axis=0);cv=np.sum((executed[:,14:]<lo[14:])|(executed[:,14:]>hi[14:]),axis=0)
        contacts={s:max(float(z[f'{s}_{d}_force_n'].max()) for d in ('thumb','index','middle')) for s in ('left','right')}
    result=dict(status='PASS' if not mv.any() and not cv.any() and np.isfinite(measured).all() else 'FAIL',
        measured_violations_per_joint=mv.tolist(),commanded_violations_per_joint=cv.tolist(),contacts=contacts,
        fastest_train_close_frames=fast,source_event_clock_adapter_offline_pass=len(offline),common_input_audit=file_record(D/'SOURCE_CLOCK_INPUT_AUDIT.json'),
        loaded_stress_trace=file_record(folder/'event_log.npz'),component_only_not_reference_result=True,
        legacy_component_qualification=file_record(D/'COMMON_DEX3_QUALIFICATION.json'),controller=file_record(ROOT/'tools/final_paper_source_clock_dex3.py'),
        physical_trace_complete=True,postsimulation_serialization_recovery='Native int conversion only; existing complete measured trace retained, never rerun or reclassified from contact success.',
        executed_pre_serialization_fix_freeze=file_record(D/'serialization_provenance_v1/EXECUTED_COMPONENT_FREEZE.json'))
    atomic_json(p,result);print('SOURCE_CLOCK_LOADED_QUALIFICATION',result['status'],flush=True)

if __name__=='__main__':main()
