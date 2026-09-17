"""Selected target-domain ACT checkpoints in fresh, matched physical scenes."""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import time
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_npz

OLD=ROOT/'outputs/contact_coordination_hybrid/20260907T073437Z/common_control/scripted_captured'


def verify_records(rows):
    for r in rows:
        if record(r['path'])!=r:raise ValueError('Changed policy dependency: '+r['path'])


def prepare(out):
    from .generalization_gate import require
    require(out, 'ACT_interface_test')
    complete=read(out/'act_training/ACT_TRAINING_COMPLETE.json')
    if set(complete['selected'])!={'A','B'}:raise ValueError('Both selected ACT checkpoints are required')
    parity=read(out/'effective_data_audit/TARGET_DATASET_PARITY.json')
    if parity['status']!='VERIFIED_EFFECTIVE_TARGET_DATASET_PARITY':raise ValueError('Target-domain data audit incomplete')
    selections={c:record(out/'act_training'/c/'SELECTED_CHECKPOINT.json') for c in ('A','B')}
    for c,r in selections.items():
        selected=read(r['path']);verify_records(selected['files']+[selected['lineage']])
        if selected!=complete['selected'][c]:raise ValueError('Selected checkpoint changed')
    membership=read(out/'matched_datasets/MEMBERSHIP.json');frames=[]
    for rows in membership['conditions'].values():
        for row in rows:
            a=read(row['alignment']['path']);frames.append(len(a['images']))
    horizon=int(np.ceil(max(frames)/30*1.5/5)*5*30)
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json');natural=read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad']
    scene=read(out/'DEV35_scenes/MANIFEST.json');split=read(out/'SPLIT_CONTRACT.json')
    assert scene['ordered_source_ids']==split['evaluation_source_ids'] and len(scene['rows'])==35
    dependencies=[*selections.values(),record(out/'effective_data_audit/TARGET_DATASET_PARITY.json'),record(out/'matched_datasets/MEMBERSHIP.json'),record(out/'DEV35_scenes/MANIFEST.json')]
    dependencies += [r['source_scene'] for r in scene['rows']]
    dependencies += [record(ROOT/'tools/contact_coordination'/n) for n in ('act_physics.py','act_policy_runtime.py','score_policy.py','policy_study.py','g1_rgb_observer.py','calibration_capture.py','runtime_hulls.py')]
    dependencies += [record(ROOT/p) for p in ('tools/act_b_inference_worker.py','tools/run_act_b_isaac_diagnostic.py','tools/common_deployment_safety_projection.py','tools/common_jerk_limited_otg.py',
        'outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json','outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json','outputs/policy_b_isaac_validation/camera/source_like_cam_high.json')]
    dependencies += read(OLD/'DEPENDENCIES.json')['files']
    dependencies += [record(out/'target_repair/runtime_bin150'/n) for n in ('RUNTIME_HULLS.json','RUNTIME_HULL_VERTICES.npz')]
    schedule=[]
    for i,sid in enumerate(split['evaluation_source_ids']):
        for c in (('A','B') if i%2==0 else ('B','A')):schedule.append(dict(index=len(schedule),condition=c,source_id=sid))
    result=dict(status='PREPARED_TARGET_DOMAIN_ACT_PROTOCOL',dependencies=dependencies,selected=selections,schedule=schedule,
        common_initial_q_rad=np.r_[natural,cal['contacts']['pregrasp']['commanded_finger_q']],joint_names=cal['joint_names'],
        horizon_control_frames=horizon,horizon_basis='1.5 times longest complete matched TRAIN demonstration, rounded up to5seconds; common to both policies.',
        control_hz=30,chunk_length=50,executed_actions_per_chunk=50,temporal_ensemble=False,execution='e0',
        randomness='No scene perturbation. Official ACT evaluation mode with deterministic zero latent and reset action buffers. Identical fixed physical scene per pair.',
        source_reference_or_event_clock_allowed=False,high_level_converter_enabled=False,
        policy_safety='Same named hard/deployment projector and persistent Ruckig OTG; actual measured state enters ACT. Actual object state is privileged collision-safety input only. No task controller or arm rescue.',
        policy_failure_retry=False,common_infrastructure_failure='Unresolved unknown; preserve and repair/version symmetrically if affected.',
        placement='Existing outcome-centric valid-bin release and >=1second settle. Opening-associated and premature-in-bin releases labeled separately.',
        source_status='DEV35 development evaluation, not untouched testing',prototype_source_id=read(out/'bootstrap/SELECTION.json')['prototype_source_id'])
    path=out/'ACT_policy/PROTOCOL.json'
    if path.exists():
        if read(path)!=__import__('json').loads(__import__('json').dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v)):raise ValueError('Immutable policy protocol changed')
    else:atomic_json(path,result)
    return result


def run_case(out,condition,source_id,folder,frames,primary=False):
    from .generalization_gate import require
    require(out, 'ACT_interface_test')
    protocol=read(out/'ACT_policy/PROTOCOL.json');verify_records(protocol['dependencies'])
    if (folder/'PROCESS.json').exists():
        verify_records(read(folder/'ACT_RUNTIME.json')['dependencies']);return folder
    if folder.exists():raise ValueError('Interrupted policy attempt requires diagnosis; no success-selected restart: '+str(folder))
    selected=read(protocol['selected'][condition]['path']);checkpoint=Path(selected['checkpoint']);q0=np.asarray(protocol['common_initial_q_rad']);names=protocol['joint_names']
    folder.mkdir(parents=True);workspace=folder/'geometry_workspace';shutil.copytree(out/'target_repair/runtime_bin150',workspace/'target_repair/runtime_bin150')
    dependencies=protocol['dependencies']+selected['files'];physics=dict(read(OLD/'DEPENDENCIES.json'));physics['files']=dependencies
    atomic_json(folder/'DEPENDENCIES.json',physics)
    cfg=dict(runtime_mode='CURRENT_RGB_MEASURED_STATE_TO_ACT_TO_COMMON_OTG',output_dir=str(folder),checker_dir=str(workspace),condition=condition,
        checkpoint=str(checkpoint),model_sha256=record(checkpoint/'model.safetensors')['sha256'],selected_checkpoint_record=protocol['selected'][condition],
        execution='e0',control_fps=30,maximum_control_frames=frames,initial_q_rad=q0,joint_names=names,
        camera=str(ROOT/'outputs/policy_b_isaac_validation/camera/source_like_cam_high.json'),
        deployment_projection=str(ROOT/'outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json'),
        otg=str(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json'),dependencies=dependencies,main_study_eligible=primary)
    atomic_json(folder/'ACT_RUNTIME.json',cfg)
    atomic_npz(folder/'ADMINISTRATIVE_LOOP.npz',commanded_q_rad=np.tile(q0,(frames,1)),joint_names=np.asarray(names),stage=np.repeat('POLICY',frames),control_fps_hz=np.asarray(30.))
    if primary:shutil.copy2(out/'DEV35_scenes'/source_id/'SOURCE_SCENE.json',folder/'SOURCE_SCENE.json')
    else:
        from .source_phase import PHYSICS
        phase=read(out/'source_phase'/source_id/'PHASE_RECORD.json');x=np.asarray(phase['initial_object_pose_world'])
        from scipy.spatial.transform import Rotation
        atomic_json(folder/'SOURCE_SCENE.json',dict(schema_version='eval35_episode_source_derived_object_registration_v1',status='PASS',physical_eval35_outcomes_read=False,one_global_canonical_object_pose=False,
            authoritative_inputs={str(PHYSICS):record(PHYSICS)['sha256']},entries=[dict(stable_episode_id=source_id,A_B_identical_object_pose=True,runtime_initialization_rule=dict(bin_pose_fixed=True),target_object_pose=dict(position_xyz_m=x[:3,3],quaternion_xyzw=Rotation.from_matrix(x[:3,:3]).as_quat()))]))
    args=list(read(OLD/'INVOCATION.json')['command']);args[1]=str(ROOT/'tools/contact_coordination/act_physics.py')
    for key,value in [('--output-dir',folder),('--direct-freeze-manifest',folder/'DEPENDENCIES.json'),('--scripted-command-path',folder/'ADMINISTRATIVE_LOOP.npz')]:args[args.index(key)+1]=str(value)
    index=args.index('--object-registration-config');args[index:index+2]=['--episode-registration-manifest',str(folder/'SOURCE_SCENE.json'),'--episode-stable-id',source_id]
    timeout=1800 if primary else 600
    atomic_json(folder/'INVOCATION.json',dict(command=args,timeout_s=timeout,source_id=source_id,condition=condition,primary_policy_attempt=primary,source_trajectory_oracle=False))
    started=time.monotonic()
    with (folder/'engine.log').open('w') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(folder/'ACTIVE_PROCESS.json',dict(pid=process.pid,command=args))
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            code='INFRASTRUCTURE_TIMEOUT'
    atomic_json(folder/'PROCESS.json',dict(returncode=code,wall_seconds=time.monotonic()-started,primary_policy_attempt=primary,complete_trace=(folder/'event_log.npz').exists()))
    return folder


def verify_interface(folder,require_physical_effect=True):
    import cv2
    cfg=read(folder/'ACT_RUNTIME.json');summary=read(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json');a=dict(np.load(folder/'event_log.npz'));f=a['control_frame']
    steps=sorted((folder/'policy_steps').glob('*.json'));queries=[];errors=[];alignment=[]
    digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    for path in steps:
        row=read(path);frame=row['control_frame'];worker=row['worker'];rgb=cv2.cvtColor(cv2.imread(row['image']['path']),cv2.COLOR_BGR2RGB)
        assert digest(rgb)==worker['rgb_sha256'] and digest(np.asarray(row['measured_state'],np.float32))==worker['state_sha256']
        expected=np.asarray(cfg['initial_q_rad']) if frame==0 else a['MEASURED_Q'][np.flatnonzero(f==frame-1)[-1]]
        alignment.append(float(np.max(np.abs(np.asarray(row['measured_state'])-expected))))
        query=frame//50*50;chunk=np.load(folder/'chunks'/f'query_{query:06d}.npz')
        np.testing.assert_allclose(row['raw_action'],chunk['physical'][frame-query],atol=1e-7,rtol=0)
        if worker['policy_query_this_call']:queries.append(frame)
        ids=np.flatnonzero(f==frame)
        if row['executed_command'] is None:assert not len(ids)
        else:
            assert len(ids)==8;errors.append(float(np.max(np.abs(a['EXECUTED_COMMAND'][ids]-np.asarray(row['executed_command'])))))
            np.testing.assert_array_equal(a['RAW_POLICY_COMMAND'][ids[0]],row['raw_action'])
    assert queries==list(range(0,len(steps),50)) and queries
    if require_physical_effect:assert len(f)>0,'TRAIN interface sanity needs an actual policy-commanded transition'
    elif not len(f):
        assert summary['aborted'] and read(folder/'ZERO_TRANSITION_POLICY_ABORT.json')['executed_control_frames']==0
        assert len(steps)==1 and read(steps[0])['executed_command'] is None
    assert summary['reset_response']['status']=='PASS' and summary['checkpoint_compatible_with_required_study']
    assert max(errors,default=0.)<1e-7 and max(alignment,default=0.)<1e-6
    assert not any(summary[k] for k in ('teacher_reference_used','source_event_clock_used','high_level_planner_used','event_driven_demo_controller_used'))
    result=dict(status='TARGET_DOMAIN_ACT_CAUSAL_INTERFACE_VERIFIED',trace=record(folder/'event_log.npz'),checkpoint=cfg['selected_checkpoint_record'],
        observation_action_records=len(steps),policy_query_frames=queries,executed_control_frames=len(np.unique(f)),
        command_alignment_error_rad=max(errors,default=0.),observation_state_alignment_error_rad=max(alignment,default=0.),
        max_executed_action_change_rad=float(np.max(np.abs(a['EXECUTED_COMMAND']-np.asarray(cfg['initial_q_rad'])))) if len(f) else None,
        actual_policy_commanded_physics_transition=bool(len(f)),
        policy_safety_abort=summary['aborted'],reference_fallback=False,task_success_required=False)
    atomic_json(folder/'INDEPENDENT_INTERFACE_VERIFICATION.json',result);return result


def zero_transition_outcome(folder):
    receipt=read(folder/'ZERO_TRANSITION_POLICY_ABORT.json')
    summary=read(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json')
    if receipt['executed_control_frames']!=0 or not summary['aborted'] or summary['executed_commands']!=0:
        raise ValueError('Unverified initial policy safety abort')
    return dict(terminal='POLICY_SAFETY_ABORT',unknown=False,full_task_success=False,
                valid_physical_rollout=False,physical_validity=None,stages={},
                first_failure='POLICY_COMMAND_SAFETY_BEFORE_FIRST_TRANSITION',
                raw_policy_abort=summary['aborted'],executed_control_frames=0,
                zero_transition_receipt=record(folder/'ZERO_TRANSITION_POLICY_ABORT.json'),
                interpretation='Actual policy inference occurred in the initialized simulation, but its unsafe command was never executed. Count as a policy-caused primary failure; no physical stage observation or valid physical rollout is fabricated.')


def sanity(out):
    protocol=prepare(out);rows=[]
    for c in ('A','B'):
        folder=out/'ACT_policy/TRAIN_sanity'/c
        run_case(out,c,protocol['prototype_source_id'],folder,151,False)
        if read(folder/'PROCESS.json')['returncode']!=0:raise RuntimeError('Policy interface process error: '+str(folder))
        rows.append(verify_interface(folder))
    result=dict(status='BOTH_TARGET_DOMAIN_ACT_INTERFACES_VERIFIED',rows=rows)
    atomic_json(out/'ACT_policy/SANITY.json',result);return result


def freeze(out):
    from .generalization_gate import require
    require(out, 'final_freeze')
    protocol=read(out/'ACT_policy/PROTOCOL.json');verify_records(protocol['dependencies']);sanity=read(out/'ACT_policy/SANITY.json')
    if sanity['status']!='BOTH_TARGET_DOMAIN_ACT_INTERFACES_VERIFIED':raise ValueError('Actual policy interface not verified')
    result=dict(status='FROZEN_ACT_A35_B35',protocol=record(out/'ACT_policy/PROTOCOL.json'),sanity=record(out/'ACT_policy/SANITY.json'),
        schedule=protocol['schedule'],scheduled_per_policy=35,checkpoint_selection_from_DEV=False)
    path=out/'ACT_FINAL_FREEZE.json'
    if path.exists() and read(path)!=result:raise ValueError('Final policy freeze changed')
    atomic_json(path,result);return result


def run(out):
    from .generalization_gate import require
    require(out, 'ACT_A35_B35')
    frozen=read(out/'ACT_FINAL_FREEZE.json');verify_records([frozen['protocol'],frozen['sanity']]);protocol=read(frozen['protocol']['path'])
    path=out/'ACT_DEV35/LEDGER.json';rows=read(path)['rows'] if path.exists() else []
    from .score_policy import score
    for item in frozen['schedule']:
        if any(r['index']==item['index'] for r in rows):continue
        folder=out/'ACT_DEV35'/f"{item['index']:03d}_{item['condition']}_{item['source_id']}"
        print('ACT_DEV35_START',item['index']+1,70,item['condition'],item['source_id'],flush=True)
        run_case(out,item['condition'],item['source_id'],folder,protocol['horizon_control_frames'],True)
        process=read(folder/'PROCESS.json');row=dict(item,attempt=str(folder),process=record(folder/'PROCESS.json'))
        if process['returncode']!=0 or not process['complete_trace']:row.update(terminal='INFRASTRUCTURE_INVALID',unknown=True,full_task_success=None,valid_physical_rollout=False)
        else:
            verify_interface(folder,require_physical_effect=False)
            if (folder/'ZERO_TRANSITION_POLICY_ABORT.json').exists():row.update(zero_transition_outcome(folder))
            else:
                outcome=score(folder,folder/'geometry_workspace');row.update(outcome);row['score']=record(folder/'ACT_PHYSICAL_SCORE.json');row['unknown']=outcome['unresolved_common_infrastructure']
        rows.append(row);atomic_json(path,dict(status='ACT_DEV35_RECORDED' if len(rows)==70 else 'ACT_DEV35_RUNNING',scheduled=70,completed=len(rows),rows=rows))
        print('ACT_DEV35_DONE',len(rows),70,item['condition'],row['terminal'],flush=True)
    return read(path)
