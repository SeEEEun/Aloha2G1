"""Short natural-start TRAIN acquisition/lift integration over shared utilities."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json,atomic_npz
from .source_phase import COMMON
from .morphology_repair import extract_calibration,wrist_target,world_wrist,assign_named
from .planner import realize_phase_goals,quintic_retime
from .runtime_hulls import Checker
from .prototype import INITIAL


def build(out,calibration_path=None,goal_overrides=None,representation='INTERACTION_OURS'):
    if representation not in ('INTERACTION_OURS','WRIST_REFERENCE'):raise ValueError('Unknown target representation')
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from .planning_kinematics import G1Kinematics
    cfg=load_common_config(COMMON);g1=G1Kinematics(cfg,load_scene(cfg))
    cal=read(calibration_path) if calibration_path is not None else extract_calibration(g1)
    atomic_json(out/'target_repair/CONTACT_CALIBRATION.json',cal)
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id'];phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
    from .scientific_cache import key as cache_key
    signature=cache_key(out,sid,'acquisition_phase_ik_and_retiming',dict(representation=representation,goal_overrides=goal_overrides))
    folder=out/'prototype'/sid/'morphology_acquisition_v4'
    receipt=folder/'CACHE_CONTRACT.json'
    if receipt.exists():
        cache=read(receipt)
        if cache['key']!=signature:raise ValueError('Changed acquisition inputs require a new immutable episode context')
        if (folder/'RESULT.json').exists():
            for artifact in cache.get('artifacts',[]):
                if record(artifact['path'])!=artifact:raise ValueError('Changed acquisition cache result')
            return read(folder/'RESULT.json')
        raise ValueError('Incomplete acquisition cache; preserve and diagnose')
    if (folder/'RESULT.json').exists():raise ValueError('Legacy acquisition cache lacks scientific dependency contract')
    atomic_json(receipt,dict(key=signature,stage='acquisition_phase_ik_and_retiming'))
    x=np.asarray(phase['initial_object_pose_world']);natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad'])
    contact=cal['contacts']['acquisition_intent'];pre=cal['contacts']['pregrasp'];held=cal['contacts']['pickup']
    openq=np.asarray(pre['commanded_finger_q']);holdq=np.asarray(contact['measured_finger_q'])
    targets={name:wrist_target(x,c) for name,c in [('PREGRASP',pre),('LEFT_ACQUISITION',contact)]}
    from .practical_parameters import parameters as practical_parameters
    practical=practical_parameters(out)
    targets['APPROACH_CLEARANCE']=targets['PREGRASP'].copy();targets['APPROACH_CLEARANCE'][2,3]+=practical['approach_clearance_m']
    targets['LIFT']=targets['LEFT_ACQUISITION'].copy();targets['LIFT'][2,3]+=.063
    checker=Checker(g1,out,cal['joint_names']);path_checks=[]
    def validate(a,b,goal):
        mode=goal['contact_mode'];n=max(2,int(np.ceil(np.max(np.abs(b-a))/.02))+1);hits=[];clearances=[]
        for i,u in enumerate(np.linspace(0,1,n)):
            q=a+u*(b-a);f=holdq if mode=='LEFT_HOLD' else openq
            predicted=x.copy()
            if mode=='LEFT_HOLD':
                assign_named(g1,np.r_[q,f],cal['joint_names']);predicted=world_wrist(g1,'left')@np.asarray(held['T_wrist_H'])@np.asarray(held['T_HO'])
            allow=() if mode=='OPEN_FREE' else ('left',)
            h=getattr(checker,'query',checker.check)(np.r_[q,f],predicted,allow,object_environment=mode=='LEFT_HOLD')
            if mode=='OPEN_FREE':
                from .incidental_contact import filter_hits
                h=filter_hits(h+checker.protected_object_contacts(np.r_[q,f],predicted),checker.incidental_policy,predicted)
            hits.extend([dict(sample=i,**v) for v in h if not v['allowed_contact']])
            if goal.get('_quality_clearance'):clearances.append(checker.clearance(np.r_[q,f],predicted,allow,object_environment=mode=='LEFT_HOLD'))
        if goal['name']=='LEFT_ACQUISITION':
            from .contact_transition_geometry import check_giver_closing
            closing=check_giver_closing(checker,b,openq,x)
            hits.extend(closing['forbidden_contacts'])
        return dict(valid=not hits,samples=n,forbidden_contacts=hits[:20],forbidden_count=len(hits),
            finger_prediction='calibrated measured HOLD' if mode=='LEFT_HOLD' else 'qualified OPEN',
            allowed_object_digits=list(allow),minimum_clearance_m=min(clearances) if clearances else None)
    goals=[dict(name=name,active_hands=['left'],wrist_pose_world={'left':targets[name]},position_tolerance_m=.003,orientation_tolerance_rad=.05,
                contact_mode='LEFT_HOLD' if name=='LIFT' else 'OPEN_FREE' if name in ('APPROACH_CLEARANCE','PREGRASP') else 'OPEN_ACQUIRE',
                cartesian_connection_steps=6)
           for name in ['APPROACH_CLEARANCE','PREGRASP','LEFT_ACQUISITION','LIFT']]
    if goal_overrides is not None:
        if set(goal_overrides)!={goal['name'] for goal in goals}:raise ValueError('Incomplete explicit acquisition goal inputs')
        for goal in goals:
            goal.update(goal_overrides[goal['name']]);targets[goal['name']]=np.asarray(goal['wrist_pose_world']['left'])
    from .source_motion_prior import attach
    goals=[attach(goal,out/'source_phase'/sid) for goal in goals]
    for goal in goals:
        goal['motion_class']='CONSTRAINED_LOCAL_CONTACT_MOTION' if goal['name']=='LEFT_ACQUISITION' else 'FREE_SPACE'
        goal['protected_object']=goal['name'] in ('APPROACH_CLEARANCE','PREGRASP')
    config=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=120)
    from .calibration_parameters import scaled_ik_config
    config=scaled_ik_config(out,config,'acquisition')
    r=realize_phase_goals(g1,goals,natural,config,candidate_validator=validate,seed_postures=[contact['seed_q']]);q=r.pop('q')
    if len(q)==2 and not r['phases'][0]['admissible']:
        # The open hand above the object has no contact-bearing orientation
        # requirement. Preserve the source-derived clearance position and
        # contact orientations at PREGRASP/acquisition; offer only a natural-
        # posture orientation prior for the FREE preparation waypoint.
        original=r;g1.assign(natural);alternative=dict(goals[0])
        target=targets['APPROACH_CLEARANCE'].copy();target[:3,:3]=world_wrist(g1,'left')[:3,:3]
        alternative.update(wrist_pose_world={'left':target},orientation_region='SO3',
            original_contact_orientation_prior=goals[0]['wrist_pose_world']['left'],
            orientation_region_provenance='Open-hand preparation one modeled object height above pregrasp. No object contact or object orientation is prescribed here; natural-q0 wrist orientation is a prior. Source-derived clearance position, complete hand geometry and every edge remain constrained. Subsequent pregrasp and acquisition contact orientations are unchanged.')
        retry=realize_phase_goals(g1,[alternative],natural,config,candidate_validator=validate,seed_postures=[contact['seed_q']]);newq=retry.pop('q')
        if retry['phases'][0]['admissible']:
            goals[0]=alternative
            tail=realize_phase_goals(g1,goals[1:],newq[-1],config,candidate_validator=validate,seed_postures=[contact['seed_q']]);tailq=tail.pop('q')
            q=np.vstack([newq,tailq[1:]]);r=dict(tail,phases=retry['phases']+tail['phases'],
                free_preparation_retry=retry,original_contact_orientation_preparation=original)
        else:r['free_preparation_retry']=retry
    folder=out/'prototype'/sid/'morphology_acquisition_v4'
    atomic_json(folder/'GOALS.json',dict(goals=goals,representation=representation,explicit_spatial_goals_supplied=goal_overrides is not None,source=record(out/'source_phase'/sid/'PHASE_RECORD.json'),calibration=record(out/'target_repair/CONTACT_CALIBRATION.json'),config=config,
        clearances=dict(hover_m=practical['approach_clearance_m'],hover_basis='common bounded preparation height; default one existing visual object height',lift_m=.063,lift_basis='qualified commanded lift primitive'),
        note='Short phase integration diagnostic, not a full-task candidate or result. Source initial position/yaw and giver role determine acquisition; contact patch is target-calibrated, source patch remains inferred/unknown.'))
    atomic_npz(folder/'PHASE_Q.npz',q=q);atomic_json(folder/'PLAN_RESULT.json',r)
    if len(q)!=5 or not all(p['admissible'] for p in r['phases']):
        result=dict(status='NO_VALID_ACQUISITION_CONNECTION',physical_run=False,phase_count=len(q)-1)
        atomic_json(folder/'RESULT.json',result)
        atomic_json(receipt,dict(key=signature,artifacts=[record(folder/n) for n in ('GOALS.json','PHASE_Q.npz','PLAN_RESULT.json','RESULT.json')]))
        return result
    bounds=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][:14]
    velocity=np.asarray([j['max_velocity_rad_s'] for j in bounds]);acceleration=np.asarray([j['max_acceleration_rad_s2'] for j in bounds])
    commands=[];labels=[];intents=[];timing=[]
    def segment(a,b,label,intent,minimum_s=0.,connecting_q=None):
        knots=np.asarray(connecting_q) if connecting_q is not None else np.asarray([a,b])
        if not (np.allclose(knots[0],a,atol=1e-12,rtol=0) and np.allclose(knots[-1],b,atol=1e-12,rtol=0)):
            raise ValueError('Acquisition connection endpoints differ from selected phases')
        pts,durations=quintic_retime(knots,velocity,acceleration)
        if len(knots)==2 and minimum_s>durations[0]:
            frames=int(np.ceil(minimum_s*30));u=np.arange(frames+1)/frames;s=10*u**3-15*u**4+6*u**5;pts=a+s[:,None]*(b-a)
        if commands:pts=pts[1:]
        commands.extend(np.c_[pts,np.tile(openq,(len(pts),1))]);labels.extend([label]*len(pts));intents.extend([intent]*len(pts));timing.append(dict(label=label,frames=len(pts),duration_s=len(pts)/30))
    connections=[p.get('connecting_q') for p in r['phases']]
    from .execution_timing import common_primitive
    primitive=common_primitive()
    segment(q[0],q[1],'APPROACH_CLEARANCE','OPEN_INTENT',connecting_q=connections[0])
    segment(q[1],q[2],'PREGRASP','OPEN_INTENT',1.5,connections[1])
    segment(q[2],q[3],'ACQUISITION_INGRESS','OPEN_INTENT',1.5,connections[2])
    segment(q[3],q[3],'PRESHAPE','LEFT_CLOSE_INTENT',primitive.preshape_frames/30.)
    segment(q[3],q[3],'POWER_GRASP','LEFT_CLOSE_INTENT',primitive.close_frames/30.)
    segment(q[3],q[3],'GRAVITY_RETENTION','LEFT_HOLD_INTENT',1.)
    segment(q[3],q[4],'LIFT_5CM','LEFT_HOLD_INTENT',1.5,connections[3])
    segment(q[4],q[4],'HOLD_ELEVATED','LEFT_HOLD_INTENT',1.)
    segment(q[4],q[3],'LOWER','LEFT_HOLD_INTENT',1.5)
    segment(q[3],q[3],'RELEASE','FINAL_RELEASE_INTENT',1.5)
    segment(q[3],q[2],'POST_RELEASE','FINAL_RELEASE_INTENT',1.5)
    commands=np.asarray(commands);command_path=folder/'COMMANDS.npz'
    # Finger commands are generated by the unchanged event controller; raw arm
    # path is continuous and starts exactly at the common natural q0.
    atomic_npz(command_path,commanded_q_rad=commands,raw_policy_command=commands,joint_names=np.asarray(cal['joint_names']),
        common_task_intent=np.asarray(intents),stage=np.asarray(labels),common_initial_q_rad=np.r_[natural,openq],
        control_fps_hz=np.asarray(30.),stable_episode_id=np.asarray(sid),method=np.asarray('a' if representation=='WRIST_REFERENCE' else 'b'),qualification_only=np.asarray(True),policy_used=np.asarray(False))
    reg=dict(schema_version='eval35_episode_source_derived_object_registration_v1',status='PASS',physical_eval35_outcomes_read=False,
        one_global_canonical_object_pose=False,authoritative_inputs={str(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json'):record(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['sha256']},
        entries=[dict(stable_episode_id=sid,A_B_identical_object_pose=True,runtime_initialization_rule=dict(bin_pose_fixed=True),
            target_object_pose=dict(position_xyz_m=x[:3,3],quaternion_xyzw=Rotation.from_matrix(x[:3,:3]).as_quat()),source_phase=record(out/'source_phase'/sid/'PHASE_RECORD.json'))])
    atomic_json(folder/'SOURCE_SCENE.json',reg)
    result=dict(status='ACQUISITION_PLAN_BUILT_PENDING_FINAL_VALIDATION',frames=len(commands),duration_s=len(commands)/30,command=record(command_path),timing=timing,
        natural_start=True,full_task=False,physical_run=False,retiming_bounds=record(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json'))
    atomic_json(folder/'RESULT.json',result)
    atomic_json(receipt,dict(key=signature,artifacts=[record(folder/n) for n in ('GOALS.json','PHASE_Q.npz','PLAN_RESULT.json','RESULT.json','COMMANDS.npz','SOURCE_SCENE.json')]))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--calibration',type=Path);a=p.parse_args();print(build(a.run_dir,a.calibration))
