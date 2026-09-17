"""Read-only source fidelity, timing and measured contact-relation diagnostics."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read, record, atomic_json


def pose_error(a,b):
    return dict(position_mm=float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),
                orientation_rad=float(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()))


def attempt_metrics(out, plan, folder=None):
    from .source_phase import COMMON,mean_pose,pose
    from .prototype import CONFIG
    from .wrist_reference import source_goals
    from .morphology_repair import assign_named,assign_measured,world_wrist
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    a=dict(np.load(plan/'COMMANDS.npz'));sid=str(a['stable_episode_id'])
    source=out/'source_phase'/sid;phase=read(source/'PHASE_RECORD.json')
    priors=dict(np.load(source/'SOURCE_PRIORS.npz'));targets=source_goals(source,read(CONFIG)['target'])
    goals={g['name']:g for g in targets['phase_goals']};c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c))
    nominal=[]
    specs=[('PREGRASP','left','pregrasp'),('POWER_GRASP','left','acquisition'),
           ('HOLD_ELEVATED','left','lift_clearance'),('DUAL_SUPPORT','left','handoff'),
           ('DUAL_SUPPORT','right','handoff'),('RIGHT_HOLD_OVER_BIN','right','right_transport'),
           ('RIGHT_PRE_RELEASE_STABILIZATION','right','placement')]
    for stage,side,source_key in specs:
        indices=np.flatnonzero(a['stage']==stage)
        if not len(indices):continue
        index=int(indices[-1]);assign_named(g,a['commanded_q_rad'][index],a['joint_names'])
        prior=mean_pose([goals['handoff_'+str(i)]['wrist_pose_world'][side] for i in range(3)]) if source_key=='handoff' else np.asarray(goals[source_key]['wrist_pose_world'][side])
        nominal.append(dict(stage=stage,side=side,command_frame=index,
                            **pose_error(world_wrist(g,side),prior)))
    selected=read(plan/'CONTACT_SELECTION.json');raw_handoff=mean_pose(priors['inferred_object_from_left'][phase['handoff_sample_indices']])
    planned_x=np.asarray(selected['objects']['left'])
    source_duration=float(priors['source_timestamp'][-1]-priors['source_timestamp'][0]);command_duration=len(a['stage'])/30
    result=dict(source_id=sid,plan=record(plan/'PLAN.json'),source_phase=record(source/'PHASE_RECORD.json'),
                functional_wrist_prior=record(source/'FUNCTIONAL_WRIST_PRIORS.npz'),
                nominal_phase_endpoint_source_wrist_deviation=nominal,
                fidelity_definition='Named model-wrist FK at phase endpoints versus registered source functional-wrist priors. Natural q0 is excluded. Command FK is nominal unloaded geometry; measured contact metrics below use full measured articulation. Fidelity is a diagnostic, never a dense10mm execution gate.',
                handoff_relocation_from_inferred_source_prior=pose_error(planned_x,raw_handoff),
                planned_handoff_object_pose=planned_x,raw_inferred_source_handoff_pose=raw_handoff,
                source_duration_s=source_duration,command_duration_s=command_duration,
                command_to_source_time_ratio=command_duration/source_duration,
                measured_contact_relation_statistics=None)
    if folder is not None:
        trace=dict(np.load(folder/'event_log.npz'));frames=trace['control_frame'];ix=np.r_[np.flatnonzero(np.diff(frames)!=0),len(frames)-1]
        contacts=selected.get('carry_contacts',selected['contacts']);overlap=selected.get('overlap_contacts',selected['contacts'])
        values={'left':[],'right':[],'dual':[]}
        for i in ix:
            supported={s:trace[s+'_thumb_force_n'][i]>=.015 and max(trace[s+'_index_force_n'][i],trace[s+'_middle_force_n'][i])>=.015 for s in ('left','right')}
            if not any(supported.values()) or trace['table_contact_force_n'][i]>.02:continue
            assign_measured(g,trace,int(i));actual=pose(Rotation.from_quat(trace['object_quaternion_xyzw'][i]).as_matrix(),trace['object_position_world_m'][i]);predicted={}
            for side in ('left','right'):
                relation=(overlap if all(supported.values()) else contacts)[side]
                predicted[side]=world_wrist(g,side)@np.asarray(relation['T_wrist_H'])@np.asarray(relation['T_HO'])
                if supported[side]:values[side].append(pose_error(predicted[side],actual))
            if all(supported.values()):values['dual'].append(pose_error(predicted['left'],predicted['right']))
        statistics={}
        for key,rows in values.items():
            statistics[key]=dict(sampled_control_frames=len(rows))
            for metric in ('position_mm','orientation_rad'):
                v=[row[metric] for row in rows]
                statistics[key][metric]=dict(median=float(np.median(v)),maximum=float(np.max(v))) if v else None
        controller=read(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json')
        event_frames=controller.get('events',{})
        event_seconds={k:(v/30 if v is not None else None) for k,v in event_frames.items()
                       if k.endswith('_frame') and (v is None or isinstance(v,int))}
        for k in ('grasp_confirmed_frames','opposing_enclosure_candidate_frames','lift_start_frames','table_support_loss_frames'):
            if k in event_frames:event_seconds[k]={side:(v/30 if v is not None else None) for side,v in event_frames[k].items()}
        executed=trace['EXECUTED_COMMAND'][ix]
        stationary=float(np.count_nonzero(np.max(np.abs(np.diff(executed[:,:14],axis=0)),axis=1)<1e-10)/30)
        result.update(trace=record(folder/'event_log.npz'),actual_executed_control_frames=len(ix),
                      actual_execution_time_s=len(ix)/30,
                      controller_event_times_s=event_seconds,
                      raw_controller_events=event_frames,
                      stationary_arm_command_duration_s=stationary,
                      stationary_duration_definition='Consecutive30Hz executed arm commands equal within1e-10rad; includes preshape/closure, support verification and settling. It is not all free waiting, and does not imply the measured arm is perfectly stationary.',
                      measured_contact_relation_statistics=statistics,
                      measured_relation_definition='30Hz final-substep samples with actual opposing hand contact and no table support. Predicted object poses use selected target contact transforms and full measured articulation. Deviations are target contact consistency, not observed ALOHA force or object-pose accuracy. Dual samples require both actual hand contacts.',
                      phase_controller_summary=record(folder/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json'))
        stop=folder/'PHASE_STOP.json'
        if stop.exists():result['phase_stop']=read(stop)
    return result


def run(out):
    rows=[];milestone=read(out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json')
    rows.append(dict(scope='TRAIN_DEVELOPMENT_MILESTONE',**attempt_metrics(out,Path(milestone['plan']['path']).parent,Path(milestone['trace']['path']).parent)))
    for scope in ('TRAIN40_conversion','reference_coupling10'):
        ledger=out/scope/'LEDGER.json'
        if not ledger.exists():continue
        for row in read(ledger)['rows']:
            if not row.get('full_task_plan'):continue
            values=attempt_metrics(out,Path(row['plan']),Path(row['attempt']) if row.get('physical_run') else None)
            rows.append(dict(scope=scope,condition=row['condition'],**values))
    result=dict(status='READ_ONLY_MECHANISTIC_EVIDENCE_RECORDED',rows=rows,implementation=record(__file__))
    atomic_json(out/'MECHANISTIC_METRICS.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    print(run(p.parse_args().run_dir)['status'])
