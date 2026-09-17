"""Recover simultaneous dual-support contacts separately from sole-hand carry."""
from pathlib import Path
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json
from .source_phase import COMMON,pose,mean_pose
from .morphology_repair import assign_measured,world_wrist


def build(out,trace_folder):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.finalize_common_dex3_grasp_qualification import CONTACT_THRESHOLD_N,TABLE_THRESHOLD_N
    path=out/'target_repair/CONTACT_CALIBRATION.json';cal=read(path);before=out/'target_repair/CONTACT_CALIBRATION_BEFORE_PHASE_CONTACTS.json'
    if before.exists():raise FileExistsError(before)
    atomic_json(before,cal);a=dict(np.load(trace_folder/'event_log.npz'));events=read(trace_folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json')['events']
    start=events['grasp_confirmed_frames']['right'];end=events['giving_hand_release_frame'];assert start is not None and end is not None and end>start
    ids=np.flatnonzero((a['control_frame']>=start)&(a['control_frame']<end));ids=ids[len(ids)//3:]
    mask=a['table_contact_force_n'][ids]<=TABLE_THRESHOLD_N
    for side in ('left','right'):
        mask &= (a[f'{side}_thumb_force_n'][ids]>=CONTACT_THRESHOLD_N)&((a[f'{side}_index_force_n'][ids]>=CONTACT_THRESHOLD_N)|(a[f'{side}_middle_force_n'][ids]>=CONTACT_THRESHOLD_N))
    assert np.all(mask) and len(ids)>=24,'At least0.1s of simultaneous sustained opposing contacts required'
    selected=ids[::8];cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));i=int(selected[len(selected)//2]);deltas={}
    cal['contacts']['right_carry']=copy.deepcopy(cal['contacts']['handoff_right'])
    for side in ('left','right'):
        name='handoff_'+side;contact=copy.deepcopy(cal['contacts'][name]);fixed=np.asarray(contact['T_wrist_H']);values=[]
        for j in selected:
            assign_measured(g,a,j);x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][j]).as_matrix(),a['object_position_world_m'][j]);values.append(np.linalg.inv(world_wrist(g,side)@fixed)@x)
        relation=mean_pose(values);prior=np.asarray(contact['T_HO']);deltas[side]=dict(position_m=float(np.linalg.norm(relation[:3,3]-prior[:3,3])),rotation_rad=float(Rotation.from_matrix(relation[:3,:3].T@prior[:3,:3]).magnitude()))
        contact.update(T_HO=relation,rows=selected,seed_q=a['MEASURED_Q'][i,:14],measured_finger_q=a['MEASURED_Q'][i,14:28],commanded_finger_q=a['EXECUTED_COMMAND'][i,14:28],
            stage='MEASURED_SIMULTANEOUS_DUAL_SUPPORT_BEFORE_GIVER_RELEASE',evidence='MEASURED_PHASE_SPECIFIC_CONTACT_CALIBRATION',trace=record(trace_folder/'event_log.npz'),
            relation_position_spread_m=float(np.max(np.linalg.norm(np.asarray(values)[:,:3,3]-relation[:3,3],axis=1))),relation_rotation_spread_rad=float(np.max(Rotation.from_matrix(np.asarray(values)[:,:3,:3]@relation[:3,:3].T).magnitude())))
        cal['contacts'][name]=contact
    cal['phase_contact_calibration']=dict(before=record(before),trace=record(trace_folder/'event_log.npz'),frames=[int(a['control_frame'][ids[0]]),int(a['control_frame'][ids[-1]])],rows=len(ids),changes=deltas,
        reason='Sole-right retention follows an actual object pivot after giver release; that contact relation is not the receiver-acquisition relation. Recover both simultaneous contacts from the same measured dual-support interval and keep carry relations separate.',
        shared_A_B_asset=True,world_task_trajectory_copied=False,controller_changed=False)
    atomic_json(path,cal);return cal['phase_contact_calibration']


def receiver_intent(out,trace_folder):
    """Calibrate the incoming-object/command relation, distinct from retention."""
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    from tools.finalize_common_dex3_grasp_qualification import CONTACT_THRESHOLD_N,TABLE_THRESHOLD_N
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json');a=dict(np.load(trace_folder/'event_log.npz'))
    events=read(trace_folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json')['events']
    primitive=Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER))
    trigger=events['right_handoff_close_intent_frame'];confirmed=events['grasp_confirmed_frames']['right'];release=events['giving_hand_release_frame']
    before=np.flatnonzero((a['control_frame']>=trigger-primitive.right_verification_frames)&(a['control_frame']<trigger))
    after=np.flatnonzero((a['control_frame']>=confirmed+3)&(a['control_frame']<release))
    assert len(before)>=24 and len(after)>=24
    assert np.all(a['right_thumb_force_n'][before]<CONTACT_THRESHOLD_N)
    assert np.all(a['table_contact_force_n'][before]<=TABLE_THRESHOLD_N)
    assert np.all(a['left_thumb_force_n'][before]>=CONTACT_THRESHOLD_N)
    object_poses=lambda ids:np.asarray([pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i]) for i in ids])
    xpre=mean_pose(object_poses(before));xpost=mean_pose(object_poses(after));transition=np.linalg.inv(xpre)@xpost
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));contact=copy.deepcopy(cal['contacts']['handoff_right']);fixed=np.asarray(contact['T_wrist_H']);values=[]
    for i in after[::8]:
        assign_measured(g,a,i,'EXECUTED_COMMAND')
        values.append(np.linalg.inv(world_wrist(g,'right')@fixed)@xpre)
    relation=mean_pose(values);contact.update(T_HO=relation,stage='RECEIVER_PRECONTACT_OBJECT_TO_ACQUISITION_COMMAND',
        evidence='CALIBRATED_ACQUISITION_INTENT_NOT_A_RIGID_RETAINED_CONTACT',trace=record(trace_folder/'event_log.npz'),
        precontact_rows=before[::8],command_rows=after[::8],seed_q=a['EXECUTED_COMMAND'][after[len(after)//2],:14],
        relation_position_spread_m=float(np.max(np.linalg.norm(np.asarray(values)[:,:3,3]-relation[:3,3],axis=1))))
    cal['contacts']['receiver_acquisition_intent']=contact
    cal['receiver_capture_transition']=dict(T_preobject_postobject=transition,position_change_m=float(np.linalg.norm(transition[:3,3])),rotation_change_rad=float(Rotation.from_matrix(transition[:3,:3]).magnitude()),
        gravity_up_in_preobject=xpre[:3,:3].T@np.array([0.,0.,1.]),trace=record(trace_folder/'event_log.npz'),
        before_control_window=[int(a['control_frame'][before[0]]),int(a['control_frame'][before[-1]])],after_control_window=[int(a['control_frame'][after[0]]),int(a['control_frame'][after[-1]])],
        interpretation='The calibrated receiver approach/closing physically displaces the held object before dual support. Incoming object pose and post-capture object pose are separate phase variables. This relative capture model predicts geometry only; runtime object remains dynamic.',
        full_world_trajectory_exported=False,source_world_pose_copied=False,measured_contact_geometry_still_requires_dynamic_verification=True)
    cal['receiver_calibration_parent']=record(out/'target_repair/CONTACT_CALIBRATION.json')
    destination=out/'target_repair/CONTACT_CALIBRATION_WITH_RECEIVER_INTENT.json'
    if destination.exists():raise FileExistsError(destination)
    atomic_json(destination,cal);return dict(path=str(destination.resolve()),**cal['receiver_capture_transition'])


def paired_incoming(out):
    """Use the same incoming-object convention for giver and receiver commands."""
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    parent=out/'target_repair/CONTACT_CALIBRATION_WITH_RECEIVER_INTENT.json';cal=read(parent)
    receiver=cal['contacts']['receiver_acquisition_intent'];trace=Path(receiver['trace']['path'])
    assert record(trace)==receiver['trace']
    a=dict(np.load(trace));indices=np.asarray(receiver['precontact_rows'],dtype=int)
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));contact=copy.deepcopy(cal['contacts']['left_carry']);fixed=np.asarray(contact['T_wrist_H'])
    commanded=[];measured=[];compensation=[]
    for i in indices:
        x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
        assign_measured(g,a,i);hm=world_wrist(g,'left')@fixed
        assign_measured(g,a,i,'EXECUTED_COMMAND');hc=world_wrist(g,'left')@fixed
        measured.append(np.linalg.inv(hm)@x);commanded.append(np.linalg.inv(hc)@x);compensation.append(np.linalg.inv(hc)@hm)
    contact.update(T_HO=mean_pose(commanded),measured_T_HO=mean_pose(measured),T_commandHand_measuredHand=mean_pose(compensation),
        seed_q=a['EXECUTED_COMMAND'][indices[len(indices)//2],:14],trace=record(trace),rows=indices,
        stage='GIVER_INCOMING_HOLD_BEFORE_RECEIVER_CONTACT',evidence='CALIBRATED_COMMAND_REALIZATION_WITH_MEASURED_CONTACT_RECORDED_SEPARATELY',
        measured_finger_q=a['MEASURED_Q'][indices[len(indices)//2],14:],commanded_finger_q=a['EXECUTED_COMMAND'][indices[len(indices)//2],14:],
        interpretation='Same qualified pre-contact window used by receiver acquisition intent. This command-side transform includes loaded arm deflection and is a prediction, not a new fixed physical tool transform. Measured physical contact transform and commanded-to-measured hand transform are explicit.')
    cal['contacts']['giver_handoff_intent']=contact;cal['paired_incoming_parent']=record(parent)
    path=out/'target_repair/CONTACT_CALIBRATION_PAIRED_INCOMING.json'
    if path.exists():raise FileExistsError(path)
    atomic_json(path,cal);return dict(path=str(path.resolve()),contact=contact)


def current_giver(out,trace_folder):
    """Calibrate the actual established giver grasp, not a borrowed latch shape."""
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    cal=read(out/'target_repair/CONTACT_CALIBRATION_COMMAND_PHASES.json');a=dict(np.load(trace_folder/'event_log.npz'))
    primitive=Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER))
    event=read(trace_folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json')['events'];trigger=event['right_handoff_close_intent_frame']
    indices=np.flatnonzero((a['control_frame']>=trigger-primitive.right_verification_frames)&(a['control_frame']<trigger))[::8]
    from tools.finalize_common_dex3_grasp_qualification import CONTACT_THRESHOLD_N
    assert np.all(a['right_thumb_force_n'][indices]<CONTACT_THRESHOLD_N) and np.all(a['left_thumb_force_n'][indices]>=CONTACT_THRESHOLD_N)
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));c=copy.deepcopy(cal['contacts']['giver_handoff_intent']);fixed=np.asarray(c['T_wrist_H']);commanded=[];measured=[];compensation=[]
    for i in indices:
        x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
        assign_measured(g,a,i);hm=world_wrist(g,'left')@fixed
        assign_measured(g,a,i,'EXECUTED_COMMAND');hc=world_wrist(g,'left')@fixed
        measured.append(np.linalg.inv(hm)@x);commanded.append(np.linalg.inv(hc)@x);compensation.append(np.linalg.inv(hc)@hm)
    i=indices[len(indices)//2];c.update(T_HO=mean_pose(commanded),measured_T_HO=mean_pose(measured),T_commandHand_measuredHand=mean_pose(compensation),
        measured_finger_q=a['MEASURED_Q'][i,14:],commanded_finger_q=a['EXECUTED_COMMAND'][i,14:],seed_q=a['EXECUTED_COMMAND'][i,:14],trace=record(trace_folder/'event_log.npz'),rows=indices,
        evidence='CURRENT_VERIFIED_TRAIN_GIVER_CARRY_CALIBRATION',
        interpretation='The current natural source-conditioned grasp has a different contact-driven finger latch than the older giver calibration. Preserve its actual pre-receiver held relation, measured fingers and command realization; receiver capability remains independently calibrated. Shared TRAIN calibration asset, no world task trajectory copied.')
    cal['contacts']['giver_handoff_intent']=c
    destination=out/'target_repair'/('CONTACT_CALIBRATION_CURRENT_GIVER_'+record(trace_folder/'event_log.npz')['sha256'][:12]+'.json')
    if destination.exists():raise FileExistsError(destination)
    atomic_json(destination,cal);return dict(path=str(destination.resolve()))


def right_carry_command(out):
    """Reproduce the qualified sole-receiver command realization transform."""
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    cal=read(out/'target_repair/CONTACT_CALIBRATION_PAIRED_INCOMING.json');c=copy.deepcopy(cal['contacts']['right_carry'])
    assert record(c['control']['path'])==c['control']
    a=dict(np.load(c['control']['path']));cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));values=[]
    for i in c['rows']:
        assign_measured(g,a,i,'EXECUTED_COMMAND')
        x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
        values.append(np.linalg.inv(world_wrist(g,'right')@np.asarray(c['T_wrist_H']))@x)
    c['measured_T_HO']=c['T_HO'];c['T_HO']=mean_pose(values)
    c['evidence']='QUALIFIED_RIGHT_ONLY_COMMAND_REALIZATION_NOT_A_FIXED_PHYSICAL_TOOL_TRANSFORM'
    cal['contacts']['right_carry_command_intent']=c
    path=out/'target_repair/CONTACT_CALIBRATION_COMMAND_PHASES.json'
    if path.exists():
        np.testing.assert_allclose(read(path)['contacts']['right_carry_command_intent']['T_HO'],c['T_HO'],atol=1e-12,rtol=0)
    else:atomic_json(path,cal)
    return dict(path=str(path.resolve()),transform_recomputed=True)


def current_right_carry(out,trace_folder):
    """Recover an actually retained phase with its full attempt provenance."""
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.finalize_common_dex3_grasp_qualification import CONTACT_THRESHOLD_N,TABLE_THRESHOLD_N
    parent=Path(read(Path(read(out/'target_repair/CURRENT_CONTACT_REGION_FIT.json')['path'])/'CONFIG.json')['calibration']['path'])
    cal=read(parent);a=dict(np.load(trace_folder/'event_log.npz'));score=read(trace_folder/'HYBRID_SCORE.json')
    if not score['physical_validity']:raise ValueError('A current carry qualification requires a physically valid measured attempt')
    selected=read(trace_folder.parent/'SELECTED_CONTACTS.json')
    right=(a['right_thumb_force_n']>=CONTACT_THRESHOLD_N)&((a['right_index_force_n']>=CONTACT_THRESHOLD_N)|(a['right_middle_force_n']>=CONTACT_THRESHOLD_N))
    giver=np.max(np.column_stack([a['left_'+d+'_force_n'] for d in ('thumb','index','middle','palm')]),axis=1)>=CONTACT_THRESHOLD_N
    mask=right&~giver&(a['table_contact_force_n']<=TABLE_THRESHOLD_N)&(a['stage']=='DUAL_SUPPORT')
    runs=np.split(np.flatnonzero(mask),np.flatnonzero(np.diff(np.flatnonzero(mask))!=1)+1)
    ids=max(runs,key=len);assert len(ids)>=240,'A full second of measured sole-receiver opposing support is required'
    # Central one-second window selected mechanically, not a final successful run.
    offset=(len(ids)-240)//2;indices=ids[offset:offset+240:8]
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));c=copy.deepcopy(cal['contacts']['right_carry_command_intent']);fixed=np.asarray(c['T_wrist_H'])
    commanded=[];measured=[];gravity=[]
    for i in indices:
        x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
        assign_measured(g,a,i);hm=world_wrist(g,'right')@fixed
        assign_measured(g,a,i,'EXECUTED_COMMAND');hc=world_wrist(g,'right')@fixed
        commanded.append(np.linalg.inv(hc)@x);measured.append(np.linalg.inv(hm)@x);gravity.append(x[:3,:3].T@np.array([0.,0.,1.]))
    i=indices[len(indices)//2]
    c.update(T_HO=mean_pose(commanded),measured_T_HO=mean_pose(measured),measured_finger_q=a['MEASURED_Q'][i,14:],commanded_finger_q=a['EXECUTED_COMMAND'][i,14:],seed_q=a['EXECUTED_COMMAND'][i,:14],
        rows=indices,control=record(trace_folder/'event_log.npz'),evidence='CURRENT_PHYSICALLY_VALID_TRAIN_SOLE_RECEIVER_PHASE_CALIBRATION',
        acquisition_reference_T_HO=selected['contacts']['right']['T_HO'],acquisition_selection=record(trace_folder.parent/'SELECTED_CONTACTS.json'),
        attempt_score=record(trace_folder/'HYBRID_SCORE.json'),full_task_demonstrated=score['source_conditioned_full_task']=='DEMONSTRATED',
        gravity_up_in_object=np.mean(gravity,axis=0),retention_window_s=len(ids)/240.,
        interpretation='The dynamic object is retained with no giver/table support in this measured interval. The complete attempt score and all later outcomes remain linked. This local command-realization prediction is not an attachment, task trajectory, or full-task success.')
    cal['contacts']['right_carry_command_intent']=c
    destination=out/'target_repair'/('CONTACT_CALIBRATION_CURRENT_CARRY_'+record(trace_folder/'event_log.npz')['sha256'][:12]+'.json')
    if destination.exists():raise FileExistsError(destination)
    atomic_json(destination,cal);return dict(path=str(destination.resolve()),retained_interval_s=len(ids)/240.)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--trace-folder',type=Path);p.add_argument('--receiver-intent',action='store_true');p.add_argument('--paired-incoming',action='store_true');p.add_argument('--right-carry-command',action='store_true');p.add_argument('--current-giver',action='store_true');p.add_argument('--current-right-carry',action='store_true');a=p.parse_args();print(current_right_carry(a.run_dir.resolve(),a.trace_folder.resolve()) if a.current_right_carry else current_giver(a.run_dir.resolve(),a.trace_folder.resolve()) if a.current_giver else right_carry_command(a.run_dir.resolve()) if a.right_carry_command else paired_incoming(a.run_dir.resolve()) if a.paired_incoming else (receiver_intent if a.receiver_intent else build)(a.run_dir.resolve(),a.trace_folder.resolve()))
