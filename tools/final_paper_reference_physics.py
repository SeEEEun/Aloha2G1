#!/usr/bin/env python3
"""Frozen source-clock command preparation and direct reference PhysX cohort."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_run import OUT,DEST,FREEZE,RESET,SPLIT,read,file_record,atomic_json,atomic_npz,atomic_text,cases,now
from tools.common_execution_layer import Dex3Primitive
from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
from tools.direct_physical_execution_layer import authoritative_joint_limits
JOINT=ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json'
REG=ROOT/'outputs/final_episode_registered_eval35/00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json'
PF=OUT/'03_common_execution_freeze/COMMON_REFERENCE_PHYSICS_FREEZE_MANIFEST.json'
ISAAC='/home/jbnu/miniconda3/envs/isaaclab6/bin/python'
STAGES=['EXECUTABLE_TRAJECTORY','APPROACH_VALID','LEFT_GRASP_SUCCESS','LIFT_SUCCESS','HANDOFF_SUCCESS','RIGHT_OWNERSHIP_SUCCESS','RIGHT_TRANSPORT_SUCCESS','BIN_ENTRY_SUCCESS','BIN_SETTLE_SUCCESS','FULL_TASK_SUCCESS']

def nominal_from_source(raw):
    """Common primitive shapes, unchanged source event clock and close signal."""
    primitive=Dex3Primitive.from_frozen_dependencies(read(PHYSICS_CONFIG),read(PHYSICAL_ENVIRONMENT),read(COMMON_PHYSICAL_CONTROLLER))
    lo,hi,names=authoritative_joint_limits(read(JOINT));times=raw['source_timestamp'];count=len(times)
    events=dict(zip(raw['source_event_names'].astype(str),raw['common_execution_event_times_sec']))
    # Source event times have float32 quantization while archived timestamps may
    # be float64. Nearest timestamp mapping preserves the same sampled event;
    # ceiling/searchsorted can spuriously postpone it by one full frame.
    mapped={k:int(np.argmin(np.abs(times-v))) for k,v in events.items()}
    ef={k:21+v for k,v in mapped.items()}
    ts=times-times[0];parts=[]
    for side in ('left','right'):
        opened=getattr(primitive,f'{side}_open');preshape=getattr(primitive,f'{side}_preshape');closed=getattr(primitive,f'{side}_full_close')
        fraction=raw[f'common_{side}_close_fraction']
        values=opened[None]+fraction[:,None]*(closed-opened)[None]
        begin=times[mapped['APPROACH_START'] if side=='left' else mapped['RIGHT_APPROACH_BEGIN']]
        close=times[mapped['LEFT_CLOSE_BEGIN'] if side=='left' else mapped['RIGHT_CLOSE_BEGIN']]
        end=times[mapped['LEFT_CLOSE_COMPLETE'] if side=='left' else mapped['RIGHT_ACQUIRE_SOURCE']]
        presha=(times>=begin)&(times<close)
        alpha=np.clip((times-begin)/max(close-begin,1/30),0,1)
        values[presha]=opened+alpha[presha,None]*(preshape-opened)
        closing=(times>=close)&(times<=end)
        beta=np.clip((times-close)/max(end-close,1/30),0,1)
        values[closing]=preshape+beta[closing,None]*(closed-preshape)
        parts.append(np.vstack((np.tile(opened,(21,1)),values)))
    hand=np.clip(np.hstack(parts),lo[14:]+.005,hi[14:]-.005)
    intent=np.full(count+21,'OPEN_INTENT',dtype='U24')
    for name,label in [('LEFT_CLOSE_BEGIN','LEFT_CLOSE_INTENT'),('LEFT_CLOSE_COMPLETE','LEFT_HOLD_INTENT'),('RIGHT_CLOSE_BEGIN','HANDOFF_INTENT'),('RIGHT_OWNERSHIP_SOURCE','RIGHT_HOLD_INTENT'),('FINAL_RELEASE_BEGIN','FINAL_RELEASE_INTENT')]:intent[ef[name]:]=label
    return hand,intent,ef,names

def command(row,trajectory):
    with np.load(row['source']['path']) as raw:hand,intent,ef,names=nominal_from_source(raw)
    with np.load(trajectory) as z:arms=z['full_q'];timestamps=z['execution_timestamp']
    assert len(arms)==len(hand)
    values=np.hstack((arms,hand));entry=next(x for x in read(REG)['entries'] if x['eval_index']==row['index']) if row['group']=='DEV35' else None
    path=DEST/'reference_commands'/f"{row['key']}.npz"
    atomic_npz(path,commanded_q_rad=values,raw_policy_command=values,policy_safe_command=values,common_source_nominal_dex3=hand,
        common_task_intent=intent,stage=intent,joint_names=np.asarray(names),control_fps_hz=np.asarray(30.),common_initial_q_rad=values[0],
        source_event_names=np.asarray(list(ef)),execution_event_frames=np.asarray(list(ef.values())),execution_timestamp=timestamps,
        stable_episode_id=np.asarray(entry['stable_episode_id'] if entry else row['source_recording_id']),
        runtime_right_three_digit_gate_required=np.asarray(False),policy_used=np.asarray(False),standardized_initial_grasp=np.asarray(False))
    return path,entry

def freeze_physics():
    d=read(DEST/'dex3/COMMON_DEX3_QUALIFICATION.json');assert d['status']=='PASS'
    assert read(DEST/'dex3/SOURCE_CLOCK_QUALIFICATION.json')['status']=='PASS'
    if PF.exists():return read(PF)
    old=read(DEST/'dex3/COMMON_COMPONENT_PROVISIONAL_FREEZE.json');paths={Path(r['path']) for r in old['files']}
    paths.update([Path(__file__),ROOT/'tools/final_paper_source_clock_dex3.py',ROOT/'tools/final_paper_physics_isaac.py',ROOT/'tools/final_paper_complete_action_qualification.py',ROOT/'tools/score_episode_registered_physical_eval35_run.py',ROOT/'tools/final_paper_score_physics.py',REG,FREEZE,OUT/'03_common_execution_freeze/COMMON_FULL6D_FREEZE_MANIFEST.json',OUT/'01_registration/DEV35_EPISODE_TASK_ENVIRONMENTS.json',DEST/'dex3/SOURCE_CLOCK_QUALIFICATION.json'])
    current=read(OUT/'01_registration/DEV35_EPISODE_TASK_ENVIRONMENTS.json')['entries'];oldreg=read(REG)['entries']
    for e in current:
        r=next(x for x in oldreg if x['stable_episode_id']==e['stable_episode_id'])
        assert r['target_object_pose']==e['T_object_A']==e['T_object_B']
    manifest=dict(status='FROZEN_BEFORE_EVAL35',created_at=now(),graspability_classifier_used=False,files=[file_record(p) for p in sorted(paths)],
        registration_exact_matches=35,source_event_clock_unchanged=True,nominal_source_clock_controller=True,
        physical_failures='Final data, including normal contact loss. Infrastructure failures are separately classified and repaired.',
        cumulative_denominator=35,stages=STAGES,common_scorer='Existing mechanical-contact scorer plus separate approach, grasp and lift stages; no atlas',
        hardware_used=False,prep_seconds=.7,real_object_pose_writes_after_initialization=0)
    atomic_json(PF,manifest);atomic_json(DEST/'PHYSICS_FREEZE_SHA256.json',file_record(PF));return manifest

def run_one(row):
    folder=DEST/'reference_physics'/row['key'];rp=folder/'RESULT.json'
    if rp.exists() and read(rp)['outcome']!='INFRASTRUCTURE_INVALID':return read(rp)
    position=read(DEST/'position'/row['key']/'RESULT.json');six=read(DEST/'full6d'/row['key']/'RESULT.json')
    if position['outcome']=='INFRASTRUCTURE_INVALID' or six['outcome']=='INFRASTRUCTURE_INVALID':raise RuntimeError('Resolve numerical-stage infrastructure before physics')
    if six['outcome']!='FULL6D_EXECUTABLE':
        reason=position['outcome'] if position['outcome']!='POSITION_EXECUTABLE' else six['outcome']
        r=dict(case=row,outcome='RETARGETING_FEASIBILITY_FAILURE',failure_class=reason,outcomes={s:False for s in STAGES},
            first_failure_stage='EXECUTABLE_TRAJECTORY',rollout_fabricated=False,physical_trace=None,position_freeze=file_record(FREEZE))
        atomic_json(rp,r);return r
    from tools.final_paper_complete_action_qualification import qualify_action
    complete=qualify_action(row)
    if complete['outcome']!='COMPLETE_ACTION_EXECUTABLE':
        r=dict(case=row,outcome='RETARGETING_FEASIBILITY_FAILURE',failure_class=complete['outcome'],outcomes={s:False for s in STAGES},
            first_failure_stage='EXECUTABLE_TRAJECTORY',rollout_fabricated=False,physical_trace=None,complete_action_report=file_record(DEST/'complete_action'/row['key']/'RESULT.json'))
        atomic_json(rp,r);return r
    fp=read(PF)
    for dep in fp['files']:assert file_record(Path(dep['path']))==dep
    commands,entry=command(row,six['trajectory']['path'])
    folder.mkdir(parents=True,exist_ok=True)
    invocation=[ISAAC,str(ROOT/'tools/final_paper_physics_isaac.py'),'--direct-freeze-manifest',str(PF),'--config',str(PHYSICS_CONFIG),'--side','right','--geometry','INTERMEDIATE_PLUSH_PROXY','--profile','P14','--output-dir',str(folder),'--scripted-command-path',str(commands),'--object-spawn-side','left','--episode-registration-manifest',str(REG),'--episode-stable-id',entry['stable_episode_id'],'--audit-robot-bin','--full-task-audit','--bin-height-m','0.150','--bin-rim-bevel-m','0.003','--headless']
    atomic_json(folder/'INVOCATION_MANIFEST.json',dict(command=invocation,method='REFERENCE_'+row['representation_mode'],eval_index=row['index'],stable_episode_id=entry['stable_episode_id'],episode_registration=entry,commands=file_record(commands),freeze=file_record(PF)))
    for retry in range(3):
        if all((folder/n).exists() for n in ('event_log.npz','trial_result.json','DIRECT_EXECUTION_RUNTIME_SUMMARY.json','robot_bin_contacts.npz')):break
        with (folder/f'ENGINE_{retry}_{time.time_ns()}.log').open('w') as log:
            subprocess.run(invocation,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
    scorer=[sys.executable,str(ROOT/'tools/final_paper_score_physics.py'),str(folder)]
    with (folder/'SCORER.log').open('w') as log:subprocess.run(scorer,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if not rp.exists():
        atomic_json(rp,dict(case=row,outcome='INFRASTRUCTURE_INVALID',reason='Trace or scorer incomplete after bounded infrastructure retries'))
    return read(rp)

def main():
    rows=[r for r in cases() if r['group']=='DEV35']
    while not (DEST/'FULL6D_COHORT_COMPLETE.json').exists():time.sleep(15)
    while not all((DEST/'complete_action'/r['key']/'RESULT.json').exists() for r in rows):time.sleep(15)
    executable=any(read(DEST/'full6d'/r['key']/'RESULT.json')['outcome']=='FULL6D_EXECUTABLE' for r in rows)
    if executable:
        while not (DEST/'dex3/SOURCE_CLOCK_QUALIFICATION.json').exists():time.sleep(15)
        freeze_physics()
    results=[]
    for row in rows:
        r=run_one(row);results.append(r);print('REFERENCE_EPISODE_FINAL',row['key'],r['outcome'],flush=True)
    atomic_json(DEST/'REFERENCE_PHYSICS_COMPLETE.json',dict(results=results,denominator=35,scope='DEV35 DEVELOPMENT EVALUATION',all_infrastructure_valid=all(r['outcome']!='INFRASTRUCTURE_INVALID' for r in results)))

if __name__=='__main__':main()
