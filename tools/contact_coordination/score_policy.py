"""Measured, source-clock-free task outcomes for actual ACT physics."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json
from tools.finalize_common_dex3_grasp_qualification import (
    CONTACT_THRESHOLD_N,TABLE_THRESHOLD_N,BIN_CENTER_XY_M,BIN_OPENING_XY_M,
    BIN_BOTTOM_Z_M,BIN_RIM_Z_M,authoritative_limits)


def sustained(mask,seconds,dt,after=0):
    count=0;required=int(np.ceil(seconds/dt-1e-9))
    for index in range(int(after),len(mask)):
        count=count+1 if mask[index] else 0
        if count>=required:return index-required+1,index
    return None


def measured_stages(a,dt=1/240.):
    """No reference labels, source event timestamps, or controller states."""
    pos=np.asarray(a['object_position_world_m']);n=len(pos)
    names=['LEFT_GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','RIGHT_TRANSPORT','BIN_ENTRY','BIN_SETTLE','FULL_TASK']
    if n==0:return dict(stages={k:False for k in names},event_rows={},release_classification='NO_ACQUISITION')
    forces={side:np.column_stack([a[f'{side}_{digit}_force_n'] for digit in ('thumb','index','middle')]) for side in ('left','right')}
    opposing={s:(v[:,0]>=CONTACT_THRESHOLD_N)&np.any(v[:,1:]>=CONTACT_THRESHOLD_N,axis=1) for s,v in forces.items()}
    support={s:np.maximum(v.max(axis=1),a[f'{s}_palm_force_n'])>=CONTACT_THRESHOLD_N for s,v in forces.items()}
    free=np.asarray(a['table_contact_force_n'])<=TABLE_THRESHOLD_N;frames=np.asarray(a['control_frame'])
    initial_z=float(np.median(pos[frames==frames[0],2]));events={};stage={k:False for k in names}
    def accept(name,mask,seconds,after):
        interval=sustained(mask,seconds,dt,after)
        if interval is not None:stage[name]=True;events[name]=dict(start_row=interval[0],confirmed_row=interval[1],time_s=interval[1]*dt)
        return interval
    grasp=accept('LEFT_GRASP',opposing['left'],.1,0)
    lift=accept('LIFT',opposing['left']&free&(pos[:,2]>=initial_z+.05),1.,grasp[1]) if grasp else None
    dual=accept('HANDOFF',opposing['left']&opposing['right']&free,.1,lift[1]) if lift else None
    owned=accept('RIGHT_OWNERSHIP',opposing['right']&~support['left']&free,1.,dual[1]) if dual else None
    if dual:
        events['DUAL_SUPPORT']=events.pop('HANDOFF')
        if owned:events['HANDOFF']=dict(events['RIGHT_OWNERSHIP'])
    stage['HANDOFF']=bool(dual and owned)
    progress=0.;transport=None
    if owned:
        distance=np.linalg.norm(pos[:,:2]-BIN_CENTER_XY_M,axis=1)
        mask=opposing['right']&~support['left']&free&(np.arange(n)>=owned[0])
        progress=float(distance[owned[0]]-np.min(distance[mask]))
        transport=accept('RIGHT_TRANSPORT',mask&(distance<=distance[owned[0]]-.05),dt,owned[1])
    inside=np.all(np.abs(pos[:,:2]-BIN_CENTER_XY_M)<=BIN_OPENING_XY_M/2,axis=1)&(pos[:,2]>BIN_BOTTOM_Z_M)&(pos[:,2]<BIN_RIM_Z_M)
    entry=accept('BIN_ENTRY',inside,dt,transport[1]) if transport else None
    settle_count=int(round(1/dt));speed=np.linalg.norm(a['object_linear_velocity_m_s'],axis=1)
    if entry and n>=settle_count and np.all(inside[-settle_count:]) and np.all(speed[-settle_count:]<=.02) and np.max(a['doll_bin_contact_force_n'][-settle_count:])>0:
        stage['BIN_SETTLE']=True;events['BIN_SETTLE']=dict(start_row=n-settle_count,confirmed_row=n-1,time_s=(n-1)*dt)
    release='NO_ACQUISITION' if not grasp else 'NO_RIGHT_OWNERSHIP' if not owned else 'NOT_RELEASED'
    loss_frame=None
    if owned:
        values=np.unique(frames);loss_run=0
        for frame in values[values>=frames[owned[1]]]:
            indices=np.flatnonzero(frames==frame);present=np.any(forces['right'][indices]>=CONTACT_THRESHOLD_N)
            loss_run=0 if present else loss_run+1
            if loss_run==3:loss_frame=int(frame)-2;break
        if loss_frame is not None:
            indices=np.flatnonzero(frames==loss_frame);loss=int(indices[-1]);release='PREMATURE_DROP_INTO_BIN' if inside[loss] else 'DROP_OUTSIDE_VALID_BIN_REGION'
            # Opening is a diagnostic inferred from executed policy fingers,
            # not a source event or a demonstration controller command.
            if inside[loss] and 'EXECUTED_COMMAND' in a and 'right_open_q' in a:
                commands=np.asarray(a['EXECUTED_COMMAND'])[:,21:28];opened=np.asarray(a['right_open_q'])
                before=max(0,loss-int(round(1.5/dt)));delta=(commands[before]-opened)**2-(commands[loss]-opened)**2
                if np.sum(delta[:3])>1e-6 and (np.sum(delta[3:5])>1e-6 or np.sum(delta[5:])>1e-6):release='POLICY_OPENING_RELEASE_IN_VALID_BIN_REGION'
    stage['FULL_TASK']=bool(all(stage[k] for k in names[:-1]) and release in ('PREMATURE_DROP_INTO_BIN','POLICY_OPENING_RELEASE_IN_VALID_BIN_REGION'))
    return dict(stages=stage,event_rows=events,release_classification=release,first_right_contact_loss_frame=loss_frame,
        initial_object_z_m=initial_z,right_transport_progress_m=progress,
        source_events_consumed=False,controller_labels_consumed=False,
        raw_bin_entry_observed=bool(np.any(inside)),first_failure=next((k for k in names if not stage[k]),None))


def score(folder,checker_dir):
    from .runtime_hulls import Checker
    from .source_phase import COMMON,pose
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    cfg=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json');dt=float(cfg['timing']['physics_dt_s'])
    trial=read(folder/'trial_result.json');summary=read(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json');a=dict(np.load(folder/'event_log.npz'))
    runtime=read(folder/'ACT_RUNTIME.json');a['right_open_q']=np.asarray(runtime['initial_q_rad'])[21:28]
    q=np.asarray(a['MEASURED_Q']);commands=np.asarray(a['EXECUTED_COMMAND']);lower,upper=authoritative_limits()
    raw_excess=np.maximum(np.maximum(lower-q,q-upper),0.);tolerance=np.r_[np.full(14,.002),np.full(14,1e-6)]
    measured_bad=int(np.count_nonzero(raw_excess>tolerance));command_bad=int(np.count_nonzero((commands<lower-1e-9)|(commands>upper+1e-9)))
    finite=all(np.isfinite(a[k]).all() for k in ('MEASURED_Q','EXECUTED_COMMAND','object_position_world_m','object_quaternion_xyzw','object_linear_velocity_m_s','object_angular_velocity_rad_s'))
    c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));checker=Checker(g,checker_dir,list(a['joint_names']));bad=[]
    for i in range(len(q)):
        extra=dict(zip(a['all_joint_names'],a['all_measured_q_rad'][i],strict=True))
        x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
        for h in checker.check(q[i],x,('left','right'),all_joint_state=extra):
            if h['allowed_contact']:continue
            robot_pair=all(not body.startswith('/') and body!='object' for body in h['bodies'])
            if robot_pair or h['depth_m']>.003:bad.append(dict(row=i,**h))
    with np.load(folder/'robot_bin_contacts.npz') as robot_bin:robot_pen=float(np.max(robot_bin['penetration_m'],initial=0.))
    object_pen=float(np.max(a['maximum_doll_bin_penetration_m'],initial=0.));artifact=trial['artifact_checks']
    penetration=min(2*cfg['object']['contact_offset_m'],cfg['gates']['maximum_runtime_penetration_m'])
    valid=bool(finite and not measured_bad and not command_bad and not bad and robot_pen<=penetration and object_pen<=penetration
        and artifact['initial_penetration_m']<=0 and artifact['maximum_object_com_step_m']<=cfg['gates']['maximum_object_com_step_m']
        and artifact['maximum_object_angular_speed_rad_s']<=cfg['gates']['maximum_object_angular_speed_rad_s'] and not artifact['contact_api_errors']
        and trial['object_pose_writes_during_timed_loop']==0 and not trial['prohibited_attachment_used'] and not trial['state_restoration']['used'])
    measured=measured_stages(a,dt);success=valid and measured['stages']['FULL_TASK'] and summary.get('aborted') is None
    measured['stages']['FULL_TASK']=bool(success)
    unknown=bool(artifact['contact_api_errors'])
    terminal='INFRASTRUCTURE_INVALID' if unknown else 'POLICY_TASK_SUCCESS' if success else 'POLICY_SAFETY_ABORT' if summary.get('aborted') else 'POLICY_PHYSICAL_VALIDITY_ABORT' if not valid else 'POLICY_TASK_FAILURE'
    result=dict(measured,terminal=terminal,physical_validity=valid,valid_physical_rollout=bool(valid and len(q)),full_task_success=bool(success),trace=record(folder/'event_log.npz'),
        scorer=record(__file__),policy_runtime=record(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json'),geometry_forbidden_count=len(bad),first_geometry_failures=bad[:100],
        maximum_raw_joint_limit_excess_rad=float(raw_excess.max(initial=0.)),measured_limit_violations=measured_bad,command_limit_violations=command_bad,
        maximum_object_bin_penetration_m=object_pen,maximum_robot_bin_penetration_m=robot_pen,
        unresolved_common_infrastructure=unknown,
        contract=dict(contact_N=.015,table_N=.02,lift_m=.05,retention_s=1.,dual_s=.1,transport_m=.05,settle_s=1.,settle_speed_m_s=.02),
        states_clipped=False,source_clock_used=False)
    atomic_json(folder/'ACT_PHYSICAL_SCORE.json',result);return result
