"""Thin finite physical-attempt wrapper; immutable plans and fresh processes."""
import os,subprocess,time,shutil
from pathlib import Path
import numpy as np
from .io import ROOT,ISAAC,read,record,atomic_json,atomic_npz,atomic_text,fingerprint
from .planner import quintic_retime


def qualified_receiver_lead_frames():
    """Overlap the qualified preshape/close primitive with receiver approach."""
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    from .execution_timing import common_primitive
    primitive=common_primitive()
    return primitive.preshape_frames+primitive.close_frames


def export_full(out,connection,receiver_advance_frames=None):
    if receiver_advance_frames is None:receiver_advance_frames=0
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id'];base=out/'prototype'/sid/'morphology_acquisition_v4';r=read(connection/'RESULT.json')
    chosen=next(x for x in r['results'] if x['all_admissible']);sub=connection/chosen.get('subdirectory',f"seed_{chosen['candidate_seed']}");q=np.load(sub/'PHASE_Q.npz')['q'];goals=read(sub/'GOALS.json');cal=read(out/'target_repair/CONTACT_CALIBRATION.json');contact=read(connection/'SELECTED_CONTACTS.json')
    ik=read(sub/'PHASE_IK.json')
    from .execution_timing import common_primitive
    primitive=common_primitive();release_seconds=primitive.release_frames/30.
    prior=dict(np.load(base/'COMMANDS.npz'));end=np.flatnonzero(prior['stage']=='HOLD_ELEVATED')[-1]+1
    commands=list(prior['commanded_q_rad'][:end]);stages=list(prior['stage'][:end]);intents=list(prior['common_task_intent'][:end]);timing=[]
    bounds=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][:14];v=np.asarray([x['max_velocity_rad_s'] for x in bounds]);acc=np.asarray([x['max_acceleration_rad_s2'] for x in bounds]);openhand=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q'])
    def append(a,b,stage,intent,minimum_s,connecting_q=None):
        if connecting_q is not None:
            knots=np.asarray(connecting_q)
            assert np.allclose(knots[0],a,atol=1e-9) and np.allclose(knots[-1],b,atol=1e-9)
            for first,last in zip(knots[:-1],knots[1:]):append(first,last,stage,intent,minimum_s/(len(knots)-1))
            return
        points,dt=quintic_retime(np.asarray([a,b]),v,acc)
        if dt[0]<minimum_s:
            frames=int(np.ceil(minimum_s*30));u=np.arange(frames+1)/frames;s=10*u**3-15*u**4+6*u**5;points=a+s[:,None]*(b-a)
        points=points[1:];commands.extend(np.c_[points,np.tile(openhand,(len(points),1))]);stages.extend([stage]*len(points));intents.extend([intent]*len(points));timing.append(dict(stage=stage,frames=len(points),seconds=len(points)/30))
    append(q[0],q[1],'LEFT_CARRY','LEFT_HOLD_INTENT',8.,ik['phases'][0].get('connecting_q'))
    append(q[1],q[1],'LEFT_CARRY_STABILIZE','LEFT_HOLD_INTENT',1.)
    append(q[1],q[2],'RECEIVER_APPROACH','OPEN_INTENT',6.,ik['phases'][1].get('connecting_q'))
    append(q[2],q[2],'DUAL_SUPPORT','HANDOFF_INTENT',3.5+release_seconds*(2 if r.get('giver_release_policy')!='simultaneous' else 1)+max(0,qualified_receiver_lead_frames()-receiver_advance_frames)/30.)
    append(q[2],q[2],'RIGHT_OWNERSHIP_VERIFY','RIGHT_HOLD_INTENT',2.)
    append(q[2],q[3],'RECEIVER_DEPARTURE' if r.get('receiver_departure') else 'GIVER_CLEARANCE','RIGHT_HOLD_INTENT',6. if r.get('receiver_departure') else 2.,ik['phases'][2].get('connecting_q'))
    append(q[3],q[4],'RIGHT_TRANSPORT','RIGHT_HOLD_INTENT',8.,ik['phases'][3].get('connecting_q'))
    append(q[4],q[4],'RIGHT_HOLD_OVER_BIN','RIGHT_HOLD_INTENT',1.)
    append(q[4],q[5],'PLACE','RIGHT_HOLD_INTENT',8.,ik['phases'][4].get('connecting_q'))
    append(q[5],q[5],'RIGHT_PRE_RELEASE_STABILIZATION','RIGHT_HOLD_INTENT',1.)
    append(q[5],q[5],'RIGHT_RELEASE','FINAL_RELEASE_INTENT',release_seconds)
    append(q[5],q[6],'POST_RELEASE_RETREAT','FINAL_RELEASE_INTENT',3.,ik['phases'][5].get('connecting_q'))
    delayed_digits=goals[4].get('final_release_delayed_digits',[])
    delayed_begin=len(commands)
    if delayed_digits:append(q[6],q[6],'POST_RELEASE_HAND_OPEN','FINAL_RELEASE_INTENT',release_seconds)
    append(q[6],q[6],'BIN_SETTLE','FINAL_RELEASE_INTENT',2.)
    folder=connection/('physical_plan' if receiver_advance_frames==0 else f'physical_plan_advance_{receiver_advance_frames}')
    if folder.exists():raise FileExistsError('Immutable plan already exists: '+str(folder))
    commands=np.asarray(commands);boundary=int(np.flatnonzero(np.asarray(stages)=='DUAL_SUPPORT')[0])
    if receiver_advance_frames:
        begin=boundary-int(receiver_advance_frames)
        assert stages[begin]=='RECEIVER_APPROACH'
        intents[begin:boundary]=['HANDOFF_INTENT']*(boundary-begin)
    coordinated=bool(r.get('coordinated_giver_release',False))
    release_boundary=stages.index('GIVER_CLEARANCE') if coordinated else boundary
    atomic_npz(folder/'COMMANDS.npz',commanded_q_rad=commands,raw_policy_command=commands,common_task_intent=np.asarray(intents),stage=np.asarray(stages),common_initial_q_rad=prior['common_initial_q_rad'],joint_names=prior['joint_names'],control_fps_hz=np.asarray(30.),stable_episode_id=np.asarray(sid),method=np.asarray(str(prior['method'])),qualification_only=np.asarray(True),policy_used=np.asarray(False),giver_release_not_before_frame=np.asarray(release_boundary),giver_release_policy=np.asarray(r.get('giver_release_policy','simultaneous')),receiver_transition_frames=np.asarray(qualified_receiver_lead_frames()),coordinated_giver_release=np.asarray(coordinated),coordinated_release_frames=np.asarray(max(45,stages.count('GIVER_CLEARANCE'))),final_release_delayed_digits=np.asarray(delayed_digits,dtype='U16'),final_release_delayed_begin=np.asarray(delayed_begin))
    shutil.copy2(base/'SOURCE_SCENE.json',folder/'SOURCE_SCENE.json')
    if r.get('loaded_geometry'):contact=dict(contact,loaded_geometry=r['loaded_geometry'])
    atomic_json(folder/'CONTACT_SELECTION.json',contact)
    atomic_json(folder/'PLAN.json',dict(status='FULL_PLAN_PENDING_POST_RETIME_CHECK',source_id=sid,full_task=True,diagnostic=False,source_conditioned=True,connection=record(connection/'RESULT.json'),contact_selection=record(folder/'CONTACT_SELECTION.json'),timing=timing,duration_s=len(commands)/30,
        minimum_phase_time_basis='Arm paths retimed after planning; contact paths retimed by execution_timing using the same28-joint velocity/acceleration limits. Stationary closing and release windows include the full retimed primitive. No outcome-dependent duration tuning.',
        receiver_advance_frames=receiver_advance_frames,giver_release_not_before_frame=release_boundary,coordinated_giver_release=coordinated,
        implementation=record(__file__),commands=record(folder/'COMMANDS.npz')))
    return folder


def validate_full(out,plan):
    from .source_phase import COMMON
    from .runtime_hulls import Checker
    from .morphology_repair import assign_named,world_wrist
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    selection=plan/'CONTACT_SELECTION.json' if (plan/'CONTACT_SELECTION.json').exists() else plan.parent/'SELECTED_CONTACTS.json'
    a=dict(np.load(plan/'COMMANDS.npz'));cal=read(out/'target_repair/CONTACT_CALIBRATION.json');selected=read(selection);cc=selected['contacts'];overlap_cc=selected.get('overlap_contacts',cc);c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));ch=Checker(g,out,cal['joint_names'])
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    from tools.direct_physical_execution_layer import DirectPhysicalDex3ExecutionLayer,authoritative_joint_limits
    from .execution_timing import common_primitive
    primitive=common_primitive();lo,hi,names=authoritative_joint_limits(read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json'))
    from dataclasses import replace
    receiver_frames=max(primitive.preshape_frames+primitive.close_frames,int(a.get('receiver_transition_frames',0)))
    preshape=int(np.ceil(receiver_frames*primitive.preshape_frames/(primitive.preshape_frames+primitive.close_frames)))
    primitive=replace(primitive,preshape_frames=preshape,close_frames=receiver_frames-preshape)
    nominal=DirectPhysicalDex3ExecutionLayer(primitive,a['common_task_intent'],a['raw_policy_command'],a['commanded_q_rad'],'ACT-A40',lo[:14],hi[:14],lo[14:],hi[14:]);trigger=int(np.flatnonzero(a['common_task_intent']=='HANDOFF_INTENT')[0]);nominal.right_trigger=trigger;nominal.right_start_q=nominal.right_open.copy()
    phase=read(out/'source_phase'/str(a['stable_episode_id'])/'PHASE_RECORD.json');x0=np.asarray(phase['initial_object_pose_world']);bad=[];openf=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q']);lastx=x0.copy()
    left_stages={'GRAVITY_RETENTION','LIFT_5CM','HOLD_ELEVATED','LEFT_CARRY','LEFT_CARRY_STABILIZE','RECEIVER_APPROACH','DUAL_SUPPORT'}
    right_stages={'GIVER_CLEARANCE','RECEIVER_DEPARTURE','RIGHT_OWNERSHIP_VERIFY','RIGHT_TRANSPORT','RIGHT_HOLD_OVER_BIN','PLACE','RIGHT_PRE_RELEASE_STABILIZATION'}
    final_stages={'RIGHT_RELEASE','POST_RELEASE_RETREAT','POST_RELEASE_HAND_OPEN','BIN_SETTLE'}
    final_begin=next((i for i,s in enumerate(a['stage']) if s=='RIGHT_RELEASE'),len(a['stage']))
    carry_fingers=np.asarray(selected.get('carry_contacts',{}).get('right',cal['contacts'].get('right_carry',cc['right']))['measured_finger_q'])[7:]
    coordinated=bool(a.get('coordinated_giver_release',False))
    release_frame=int(a['giver_release_not_before_frame'])
    capture=selected.get('receiver_capture_transition');capture_samples=[]
    loaded_audits=[];loaded_model=None
    if selected.get('loaded_geometry'):
        model_record=selected['loaded_geometry'];assert record(model_record['path'])==model_record;loaded_model=read(model_record['path'])
        loaded_model=dict(loaded_model,measured_T_HO=selected['carry_contacts']['right']['measured_T_HO'])
    if capture:
        from .source_phase import pose
        from scipy.spatial.transform import Rotation
        delta=np.asarray(capture['T_preobject_postobject']);rv=Rotation.from_matrix(delta[:3,:3]).as_rotvec()
        capture_samples=[pose(Rotation.from_rotvec(u*rv).as_matrix(),u*delta[:3,3]) for u in (.5,1.)]
    for i,(q,stage) in enumerate(zip(a['commanded_q_rad'],a['stage'])):
        q=q.copy();f=openf.copy();side=None
        if stage in left_stages:f[:7]=cc['left']['measured_finger_q'][:7];side='left'
        if stage in right_stages:f[7:]=selected.get('carry_contacts',{}).get('right',cal['contacts'].get('right_carry',cc['right']))['measured_finger_q'][7:];side='right'
        if stage in final_stages:
            from .phase_clock_runtime import receiver_release_target
            f[7:]=receiver_release_target(carry_fingers,openf[7:],i,final_begin,primitive.release_frames,a.get('final_release_delayed_digits',[]),int(a.get('final_release_delayed_begin',final_begin)))
        if stage=='DUAL_SUPPORT':f=np.r_[overlap_cc['left']['measured_finger_q'][:7],overlap_cc['right']['measured_finger_q'][7:]]
        if stage in ('RECEIVER_APPROACH','DUAL_SUPPORT') and i>=trigger and i<trigger+primitive.preshape_frames+primitive.close_frames:
            f[7:]=nominal._right_target(i)
        q[14:]=f;assign_named(g,q,cal['joint_names']);xp=x0.copy()
        if side:
            relation=overlap_cc[side] if stage=='DUAL_SUPPORT' else selected.get('carry_contacts',{}).get(side,cal['contacts']['left_carry'] if side=='left' else cal['contacts'].get('right_carry',cc[side]))
            xp=world_wrist(g,side)@np.asarray(relation['T_wrist_H'])@np.asarray(relation['T_HO']);lastx=xp.copy()
        elif stage in final_stages:xp=lastx
        if loaded_model and stage in right_stages and loaded_model.get('schema_version',1)>=2:
            from .loaded_contact_geometry import object_pose as loaded_object_pose
            xp=loaded_object_pose(g,q,loaded_model);lastx=xp.copy()
        if stage=='GIVER_CLEARANCE' and coordinated:
            from .phase_clock_runtime import release_geometry_fingers
            from scipy.spatial.transform import Rotation,Slerp
            release_policy=str(a.get('giver_release_policy','simultaneous'))
            carry_fingers=np.asarray(selected.get('carry_contacts',{}).get('right',cal['contacts']['right_carry'])['measured_finger_q'])[7:]
            f,alpha=release_geometry_fingers(np.asarray(overlap_cc['left']['measured_finger_q'])[:7],np.asarray(overlap_cc['right']['measured_finger_q'])[7:],openf[:7],carry_fingers,i-release_frame,int(a.get('coordinated_release_frames',primitive.release_frames)),release_policy,primitive.release_frames)
            start_x=np.asarray(selected.get('overlap_objects',selected['objects'])['left'])
            rotations=Slerp([0.,1.],Rotation.from_matrix(np.asarray([start_x[:3,:3],xp[:3,:3]])))
            xp=xp.copy();xp[:3,:3]=rotations([alpha]).as_matrix()[0];xp[:3,3]=(1-alpha)*start_x[:3,3]+alpha*xp[:3,3]
            q[14:]=f
        allowed=() if stage in ('APPROACH_CLEARANCE','PREGRASP') else ('right',) if stage in right_stages and not (stage=='GIVER_CLEARANCE' and coordinated) else ('left','right')
        if stage in left_stages-{'RECEIVER_APPROACH','DUAL_SUPPORT'}:allowed=('left',)
        if stage=='GIVER_CLEARANCE' and coordinated and str(a.get('giver_release_policy','simultaneous'))=='middle_first' and i-release_frame>=2*primitive.release_frames:allowed=('right',)
        if loaded_model and (stage in right_stages or stage=='RIGHT_RELEASE'):
            from .loaded_contact_geometry import check as check_loaded
            hits=check_loaded(ch,q,xp,allowed,loaded_model,audit=loaded_audits,verify_relation=stage!='RIGHT_RELEASE')
        else:hits=ch.check(q,xp,allowed,object_environment=stage in ('LIFT_5CM','HOLD_ELEVATED','LEFT_CARRY','LEFT_CARRY_STABILIZE','RECEIVER_APPROACH','DUAL_SUPPORT'))
        if stage in ('APPROACH_CLEARANCE','PREGRASP'):
            from .incidental_contact import filter_hits
            hits=filter_hits(hits+ch.protected_object_contacts(q,xp),ch.incidental_policy,xp)
        if stage=='RECEIVER_APPROACH' and i>=trigger:
            for delta in capture_samples:hits+=ch.check(q,xp@delta,('left','right'))
        bad.extend([dict(frame=i,stage=stage,**h) for h in hits if not h['allowed_contact']])
    spec=read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs'];lower=np.asarray([s['minimum'] for s in spec]);upper=np.asarray([s['maximum'] for s in spec]);ex=float(np.maximum(np.maximum(lower-a['commanded_q_rad'],a['commanded_q_rad']-upper),0).max())
    result=dict(status='VALID' if not bad and ex==0 else 'INVALID',frames=len(a['stage']),forbidden_count=len(bad),forbidden=bad,command_limit_excess_rad=ex,post_retime_rechecked=True,geometry=record(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json'),runtime_geometry_filters_unchanged=True,full_actual_measured_state_geometry_audit_required=True)
    if loaded_model:result['loaded_geometry_predictions']=loaded_audits;result['loaded_geometry_model']=selected['loaded_geometry']
    atomic_json(plan/'FINAL_COMMAND_VALIDATION.json',result);print('FINAL_VALIDATION',result['status'],len(bad),flush=True);return result


def launch(plan,attempt_name='physics_attempt_01',diagnostic=False,capture_observations=False):
    assert read(plan/'FINAL_COMMAND_VALIDATION.json')['status']=='VALID'
    run_dir=next((p for p in plan.parents if (p/'bootstrap/SELECTION.json').exists()),None)
    if run_dir is not None and (run_dir/'target_repair/CALIBRATION_VALIDITY.json').exists() and not diagnostic:
        if read(run_dir/'target_repair/CALIBRATION_VALIDITY.json')['status']!='VERIFIED':
            raise RuntimeError('CALIBRATION_RECAPTURE_REQUIRED: complete-articulation contact calibration must be qualified before another primary execution')
    folder=plan.parent/attempt_name
    if folder.exists():raise FileExistsError('Existing physical attempt is immutable: '+str(folder))
    old=ROOT/'outputs/contact_coordination_hybrid/20260907T073437Z/common_control/scripted_captured';deps=read(old/'DEPENDENCIES.json')
    for d in deps['files']:assert record(d['path'])['sha256']==d['sha256']
    deps['files'] += [record(p) for p in [ROOT/'tools/contact_coordination/phase_physics.py',ROOT/'tools/contact_coordination/phase_clock_runtime.py',plan/'SOURCE_SCENE.json',plan/'COMMANDS.npz']]
    wrapper=ROOT/'tools/contact_coordination/phase_physics.py'
    if capture_observations:
        wrapper=ROOT/'tools/contact_coordination/demonstration_physics.py'
        camera=ROOT/'outputs/policy_b_isaac_validation/camera/source_like_cam_high.json'
        deps['files'] += [record(p) for p in [wrapper,ROOT/'tools/contact_coordination/demonstration_observation.py',ROOT/'tools/contact_coordination/g1_rgb_observer.py',ROOT/'tools/deployment_camera_config.py',camera]]
        observation_config=plan/'DEMO_OBSERVATION.json'
        assert not observation_config.exists(),'Observation capture configuration is immutable'
        atomic_json(observation_config,dict(output_dir=str(folder.resolve()),camera=str(camera),joint_names=np.load(plan/'COMMANDS.npz')['joint_names'].tolist(),pre_action=True,rate_hz=30,diagnostic_prefix_only=diagnostic))
        deps['files'].append(record(observation_config))
    scope=read(run_dir/'bootstrap/SELECTION.json').get('context_scope','TRAIN_prototype') if run_dir else 'TRAIN_prototype'
    training_scope=scope!='REFERENCE_coupling10'
    deps['purpose']='One natural-start source-conditioned '+('TRAIN ' if training_scope else 'frozen paired reference-ablation ')+('phase-prefix diagnostic' if diagnostic else 'full-task attempt')+'; no outcome selection'
    atomic_json(folder/'DEPENDENCIES.json',deps)
    shutil.copy2(ROOT/'tools/contact_coordination/phase_physics.py',folder/'PHASE_PHYSICS_SNAPSHOT.py')
    args=read(old/'INVOCATION.json')['command'];args[1]=str(wrapper);args[args.index('--direct-freeze-manifest')+1]=str(folder/'DEPENDENCIES.json');args[args.index('--output-dir')+1]=str(folder);args[args.index('--scripted-command-path')+1]=str(plan/'COMMANDS.npz')
    i=args.index('--object-registration-config');sid=read(plan/'PLAN.json')['source_id'];args[i:i+2]=['--episode-registration-manifest',str(plan/'SOURCE_SCENE.json'),'--episode-stable-id',sid]
    atomic_json(folder/'INVOCATION.json',dict(command=args,timeout_s=1200,primary_attempt=not diagnostic,TRAIN_only=training_scope,execution_scope=scope,full_task=not diagnostic,read_only_telemetry_diagnostic=diagnostic,capture_target_observations=capture_observations));started=time.monotonic()
    try:
        with (folder/'engine.log').open('w') as log:
            p=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        result=dict(returncode=p.returncode,wall_seconds=time.monotonic()-started,complete_artifacts=all((folder/n).exists() for n in ['trial_result.json','event_log.npz','DIRECT_EXECUTION_RUNTIME_SUMMARY.json']))
    except subprocess.TimeoutExpired:result=dict(returncode='TIMEOUT',wall_seconds=time.monotonic()-started,status='INFRASTRUCTURE_INVALID')
    atomic_json(folder/'PROCESS.json',result);print(result,flush=True);return folder


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--connection',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();plan=export_full(a.run_dir,a.connection);r=validate_full(a.run_dir,plan)
    if a.execute and r['status']=='VALID':launch(plan)
