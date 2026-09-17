"""Thin functional-frame adapter for the existing Wrist phase generator."""
from pathlib import Path
import copy
import numpy as np
from .io import read,record,atomic_json
from .targets import make_goals


def trajectory_reference(source_folder,side,begin,end,position_resolution=.005,rotation_resolution=.05):
    """Preserve the registered source wrist curve with geometric keyframes.

    Recursive SE(3) chord error bounds determine samples. These are spatial
    references, not interaction targets or source-clock runtime commands.
    """
    from scipy.spatial.transform import Rotation,Slerp
    phase=read(source_folder/'PHASE_RECORD.json')
    data=np.load(source_folder/'FUNCTIONAL_WRIST_PRIORS.npz');times=data['source_timestamp']
    first=phase['events'][begin]['time_s'];last=phase['events'][end]['time_s']
    ids=np.flatnonzero((times>=first)&(times<=last))
    if not len(ids):return []
    values=data[side+'_wrist_world'][ids];selected={0,len(ids)-1};pending=[(0,len(ids)-1)]
    while pending:
        i,j=pending.pop()
        if j<=i+1:continue
        u=(times[ids[i:j+1]]-times[ids[i]])/(times[ids[j]]-times[ids[i]])
        interpolated=(1-u[:,None])*values[i,:3,3]+u[:,None]*values[j,:3,3]
        rotation=Slerp([0.,1.],Rotation.from_matrix([values[i,:3,:3],values[j,:3,:3]]))(u).as_matrix()
        error=np.maximum(np.linalg.norm(values[i:j+1,:3,3]-interpolated,axis=1)/position_resolution,
            Rotation.from_matrix(np.swapaxes(rotation,1,2)@values[i:j+1,:3,:3]).magnitude()/rotation_resolution)
        k=int(np.argmax(error))+i
        if error[k-i]>1. and i<k<j:selected.add(k);pending.extend([(i,k),(k,j)])
    return [dict(wrist_pose_world={side:values[i]},source_frame=int(ids[i]),source_time_s=float(times[ids[i]]),
        geometric_position_resolution_m=position_resolution,geometric_orientation_resolution_rad=rotation_resolution,
        source_input='REGISTERED_SOURCE_WRIST_TRAJECTORY') for i in sorted(selected)]


def functional_phase(phase,frame_audit):
    """Change the wrist coordinate origin without changing tool/object geometry."""
    result=copy.deepcopy(phase)
    for side in ('left','right'):
        fixed=frame_audit['transforms'][side]
        old=np.asarray(fixed['original_wrist_to_tool']);new=np.asarray(fixed['shared_wrist_to_tool'])
        result['registered_wrist_object_relations'][side]=new@np.linalg.inv(old)@np.asarray(phase['registered_wrist_object_relations'][side])
    return result


def source_goals(source_folder,config):
    phase=read(source_folder/'PHASE_RECORD.json');frames=read(source_folder/'FUNCTIONAL_FRAME_AUDIT.json')
    if frames['source_id']!=phase['source_id'] or frames['independent_hand_rebasing']:
        raise ValueError('Invalid common Wrist registration')
    priors=dict(np.load(source_folder/'FUNCTIONAL_WRIST_PRIORS.npz'))
    adjusted=functional_phase(phase,frames)
    result=make_goals(adjusted,priors,config,'WRIST_REFERENCE')
    for goal in result['phase_goals']:
        # A source prior is an optimization objective. Its error must not be a
        # surrogate for joint/path physical validity in the shared backend.
        goal.update(hard_pose_constraint=False,source_fidelity_reported_separately=True)
    result.update(functional_frame_audit=record(source_folder/'FUNCTIONAL_FRAME_AUDIT.json'),
                  source_phase=record(source_folder/'PHASE_RECORD.json'),
                  calibrated_wrist_prior=record(source_folder/'FUNCTIONAL_WRIST_PRIORS.npz'),
                  target_contact_pair_optimizer_used=False,implementation=record(__file__))
    return result


def acquisition(out,calibration_path):
    """Use the same natural approach, geometry, retiming and hand-intent export."""
    from .acquisition_plan import build
    from .prototype import CONFIG
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    targets=source_goals(out/'source_phase'/sid,read(CONFIG)['target'])
    by_name={g['name']:g for g in targets['phase_goals']}
    prior={'PREGRASP':by_name['pregrasp'],'LEFT_ACQUISITION':by_name['acquisition'],'LIFT':by_name['lift_clearance']}
    override={name:dict(wrist_pose_world={'left':g['wrist_pose_world']['left']},hard_pose_constraint=False,
                       source_time_s=g['source_time_s'],source_frame=g['source_frame']) for name,g in prior.items()}
    from .practical_parameters import parameters as practical_parameters
    hover=np.asarray(override['PREGRASP']['wrist_pose_world']['left']).copy();hover[2,3]+=practical_parameters(out)['approach_clearance_m']
    override['APPROACH_CLEARANCE']=dict(wrist_pose_world={'left':hover},hard_pose_constraint=False,
        clearance_basis='Same modeled visual-object-height preparation permission, applied to the registered Wrist spatial prior')
    for name,begin,end in [('PREGRASP','APPROACH_START','LEFT_CLOSE_BEGIN'),
                           ('LEFT_ACQUISITION','LEFT_CLOSE_BEGIN','LEFT_GRASP_SOURCE'),
                           ('LIFT','LEFT_LIFT_BEGIN','LEFT_TRANSPORT_BEGIN')]:
        override[name]['source_wrist_reference_waypoints']=trajectory_reference(out/'source_phase'/sid,'left',begin,end)
    atomic_json(out/'target_repair/WRIST_PHASE_TARGETS.json',targets)
    return build(out,calibration_path,override,'WRIST_REFERENCE')


def full_task(out,calibration_path,loaded_geometry):
    """Registered wrist priors enter the common complete-task connector."""
    from .prototype import CONFIG
    from .source_phase import mean_pose
    from .morphology_repair import wrist_target
    from .handoff_repair import carry_contact_for_candidate
    from .full_task_plan import build
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    targets=source_goals(out/'source_phase'/sid,read(CONFIG)['target'])
    source={g['name']:g for g in targets['phase_goals']}
    overlap={s:mean_pose([source['handoff_'+str(i)]['wrist_pose_world'][s] for i in range(3)]) for s in ('left','right')}
    cal=read(calibration_path)
    cc=copy.deepcopy({'left':cal['contacts']['giver_handoff_intent'],
                      'right':cal['contacts']['receiver_acquisition_intent']})
    # Shared robot contact geometry, never an Ours-selected source-world pose.
    offset=np.asarray(read(out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json')['offset_object_frame_m'])
    t=wrist_target(np.eye(4),cc['right']);t[:3,3]+=offset
    cc['right']['T_HO']=np.linalg.inv(t@np.asarray(cc['right']['T_wrist_H']))
    capture=cal.get('receiver_capture_transition');delta=np.asarray(capture['T_preobject_postobject']) if capture else np.eye(4)
    overlap_cc=copy.deepcopy(cc)
    for side in cc:
        overlap_cc[side]['T_HO']=np.asarray(cc[side]['T_HO'])@delta
        if capture and side=='right':overlap_cc[side]['measured_finger_q']=cal['contacts']['handoff_right']['measured_finger_q']
    objects={s:overlap[s]@np.asarray(cc[s]['T_wrist_H'])@np.asarray(cc[s]['T_HO']) for s in cc}
    candidate=dict(seed=0,contact_candidate=2,q=np.r_[cc['left']['seed_q'][:7],cc['right']['seed_q'][7:]],
        contacts=cc,overlap_contacts=overlap_cc,objects=objects,
        overlap_objects={s:x@delta for s,x in objects.items()},receiver_capture_transition=capture,
        carry_contacts={'left':cal['contacts']['left_carry'],
                        'right':carry_contact_for_candidate(cal['contacts'].get('right_carry_command_intent',cal['contacts']['right_carry']),cc['right'])},
        joint_seed_is_calibration_only=True,contact_pair_optimization_used=False)
    from .contact_candidate_region import translation_bank
    candidates=[]
    for index,proposal in enumerate(translation_bank(offset)):
        option=copy.deepcopy(candidate)
        right=copy.deepcopy(cal['contacts']['receiver_acquisition_intent'])
        target=wrist_target(np.eye(4),right);target[:3,3]+=proposal['offset']
        right['T_HO']=np.linalg.inv(target@np.asarray(right['T_wrist_H']))
        option['contacts']['right']=right
        option['overlap_contacts']['right']['T_HO']=np.asarray(right['T_HO'])@delta
        option['objects']['right']=overlap['right']@np.asarray(right['T_wrist_H'])@np.asarray(right['T_HO'])
        option['overlap_objects']['right']=option['objects']['right']@delta
        option['carry_contacts']['right']=carry_contact_for_candidate(cal['contacts'].get('right_carry_command_intent',cal['contacts']['right_carry']),right)
        option.update(contact_candidate=index,candidate_priority=proposal['priority'],contact_region_proposal=proposal)
        candidates.append(option)
    # Every option retains the same registered Wrist spatial goals. The common
    # connector may reject unsafe options; no paired-contact or shared-X fit is
    # run for Wrist, and no Interaction trajectory enters this bank.
    specs=[('LEFT_CARRY','left',overlap['left']),('RECEIVER_APPROACH','right',overlap['right']),
           ('RIGHT_TRANSPORT','right',source['right_transport']['wrist_pose_world']['right']),
           ('PLACE','right',source['placement']['wrist_pose_world']['right'])]
    overrides={name:dict(wrist_pose_world={side:pose},hard_pose_constraint=False,object_region_enabled=False,
        source_fidelity_reported_separately=True,source_input='Registered functional wrist/TCP spatial prior') for name,side,pose in specs}
    for name,side,begin,end in [('LEFT_CARRY','left','LEFT_TRANSPORT_BEGIN','RIGHT_ACQUIRE_SOURCE'),
        ('RECEIVER_APPROACH','right','RIGHT_APPROACH_BEGIN','RIGHT_ACQUIRE_SOURCE'),
        ('RIGHT_TRANSPORT','right','RIGHT_TRANSPORT_BEGIN','FINAL_RELEASE_BEGIN')]:
        overrides[name]['source_wrist_reference_waypoints']=trajectory_reference(out/'source_phase'/sid,side,begin,end)
    path=out/'target_repair/WRIST_FULL_TASK_INPUT.json'
    atomic_json(path,dict(targets=targets,candidate=candidate,candidates=candidates,goal_overrides=overrides,
        calibration=record(calibration_path),implementation=record(__file__),
        contact_pair_optimization_used=False,source_world_trajectory_copied=False))
    return build(out,giver_release_policy='middle_first',receiver_departure=True,
                 loaded_geometry=loaded_geometry,spatial_prior_contract=path)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-folder',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,source_goals(a.source_folder,read(a.config)))
