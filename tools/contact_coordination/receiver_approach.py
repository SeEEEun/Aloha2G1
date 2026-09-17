"""Qualified local receiver approach, expressed relative to its selected goal.

This is a shared hand-contact primitive, not a world-frame task trajectory.
It preserves the selected receiver endpoint and the common P14 hand primitive.
"""
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_npz,fingerprint
from .source_phase import COMMON,mean_pose
from .morphology_repair import assign_measured,world_wrist
from .planner import realize_phase_goals,quintic_retime


def build(out,connection,calibration_trace):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from .runtime_hulls import Checker
    from .physical_attempt import qualified_receiver_lead_frames
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
    z=dict(np.load(calibration_trace/'event_log.npz'));events=read(calibration_trace/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json')['events']
    start=events['right_handoff_close_intent_frame'];end=events['giving_hand_release_frame'];confirmed=events['grasp_confirmed_frames']['right']
    i=np.flatnonzero(z['control_frame']==start)[-1];assign_measured(g,z,i,'EXECUTED_COMMAND');pre=world_wrist(g,'right').copy()
    wrists=[]
    for j in np.flatnonzero((z['control_frame']>=confirmed+3)&(z['control_frame']<end))[::8]:
        assign_measured(g,z,j,'EXECUTED_COMMAND');wrists.append(world_wrist(g,'right').copy())
    final=mean_pose(wrists);delta=np.linalg.inv(final)@pre
    result=read(connection/'RESULT.json');candidate=next(r for r in result['results'] if r['all_admissible'])
    sub=connection/candidate['subdirectory'];q=np.load(sub/'PHASE_Q.npz')['q'];goals=read(sub/'GOALS.json')
    key,deps=fingerprint([__file__,connection/'RESULT.json',sub/'PHASE_Q.npz',calibration_trace/'event_log.npz',out/'target_repair/CONTACT_CALIBRATION.json'])
    folder=connection/('receiver_approach_'+key[:12]);cal=read(out/'target_repair/CONTACT_CALIBRATION.json')
    if folder.exists():raise FileExistsError(folder)
    target=np.asarray(goals[1]['wrist_pose_world']['right'])@delta
    goal=dict(name='RECEIVER_PRECONTACT',active_hands=['right'],wrist_pose_world={'right':target},position_tolerance_m=.003,orientation_tolerance_rad=.05)
    checker=Checker(g,out,cal['joint_names']);f=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q']);f[:7]=cal['contacts']['left_carry']['measured_finger_q'][:7]
    def valid(a,b,goal):
        bad=[];n=max(2,int(np.ceil(np.max(np.abs(b-a))/.02))+1)
        for u in np.linspace(0,1,n):
            v=a+u*(b-a);g.assign(v);c=cal['contacts']['left_carry'];x=world_wrist(g,'left')@np.asarray(c['T_wrist_H'])@np.asarray(c['T_HO'])
            bad.extend(h for h in checker.check(np.r_[v,f],x,('left','right')) if not h['allowed_contact'])
        return dict(valid=not bad,forbidden_count=len(bad),forbidden=bad[:15],samples=n)
    solved=realize_phase_goals(g,[goal],q[1],dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=120),valid,[q[2]])
    waypoint=solved.pop('q')[-1];edge=valid(waypoint,q[2],goal)
    bounds=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][:14]
    velocity=np.asarray([s['max_velocity_rad_s'] for s in bounds]);acceleration=np.asarray([s['max_acceleration_rad_s2'] for s in bounds])
    choices=[]
    for c in solved['phases'][0]['candidates']:
        if not c['admissible']:continue
        v=np.asarray(c['q']);check=valid(v,q[2],goal)
        duration=float(quintic_retime([v,q[2]],velocity,acceleration)[1][0])
        choices.append(dict(seed=c['seed_index'],q=v,edge=check,closing_edge_minimum_s=duration,
                            valid=check['valid']))
    usable=[v for v in choices if v['valid']]
    if usable:
        selected=min(usable,key=lambda c:(c['closing_edge_minimum_s'],c['seed']));waypoint=selected['q'];edge=selected['edge']
    status='VALID' if usable else 'INVALID'
    atomic_npz(folder/'WAYPOINT.npz',q=waypoint)
    atomic_json(folder/'RESULT.json',dict(status=status,connection=record(connection/'RESULT.json'),candidate=candidate,solution=solved,closing_edge_open_hand_check=edge,connecting_candidate_selection=choices,
        T_receiver_goal_precontact=delta,goal=goal,calibration_trace=record(calibration_trace/'event_log.npz'),calibration_trigger=start,
        basis='Local receiver approach relative to the selected final receiver wrist. Calibrated close begins at the precontact pose. No task world path or object pose is copied.',
        closing_frames=max(qualified_receiver_lead_frames(),int(round(selected['closing_edge_minimum_s']*30))) if usable else None,waypoint=record(folder/'WAYPOINT.npz'),dependencies=deps,
        post_retime_complete_geometry_check_still_required=True,source_goal_endpoint_unchanged=True))
    print(folder,status,flush=True);return folder


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--connection',type=Path,required=True);p.add_argument('--calibration-trace',type=Path,required=True)
    a=p.parse_args();build(a.run_dir.resolve(),a.connection.resolve(),a.calibration_trace.resolve())
