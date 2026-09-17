"""Execute only an already admissible source-conditioned post-lift prefix.

Uses the full connection's coupled contact choice and the unchanged shared
retiming/controller. No placement candidate is executed by this diagnostic.
"""
from pathlib import Path
import shutil
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_npz,fingerprint
from .planner import quintic_retime


def build(out,connection,receiver_advance_frames=None,receiver_waypoint=None,receiver_time_scale=1.,through_dual_only=False):
    if receiver_time_scale not in (1.,2.):raise ValueError('Only the predeclared original/2x TRAIN receiver timing is available')
    if receiver_waypoint and receiver_time_scale!=1.:raise ValueError('Retiming requires a newly validated waypoint schedule')
    result=read(connection/'RESULT.json')
    choices=[]
    for row in result['results']:
        sub=connection/row.get('subdirectory',f"seed_{row['candidate_seed']}")
        ik=read(sub/'PHASE_IK.json')
        if all(p['admissible'] for p in ik['phases'][:2 if through_dual_only else 3]) and not ik['giver_opening_forbidden']:
            choices.append((row,sub))
    # RESULT preserves the contact selector's cost/seed order. Use its first
    # admissible complete prefix, rather than silently preferring seed index0.
    row,sub=choices[0]
    contacts=next(c for c in read(result['contact_bank']['path']) if c['seed']==row['candidate_seed'] and c['contact_candidate']==row.get('contact_candidate',0) and c['valid_stationary_overlap'])
    if result.get('loaded_geometry'):contacts['loaded_geometry']=result['loaded_geometry']
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    base=out/'prototype'/sid/'morphology_acquisition_v4'
    from .physical_attempt import qualified_receiver_lead_frames
    lead=0 if receiver_advance_frames is None else int(receiver_advance_frames)
    if not 0<=lead<=180*receiver_time_scale:raise ValueError('Receiver closure must start within its retimed approach or at arrival')
    key,deps=fingerprint([__file__,ROOT/'tools/contact_coordination/giver_clearance.py',connection/'RESULT.json',sub/'PHASE_Q.npz',sub/'GOALS.json',base/'COMMANDS.npz',out/'target_repair/CONTACT_CALIBRATION.json',ROOT/'tools/contact_coordination/physical_attempt.py',ROOT/'tools/contact_coordination/phase_physics.py',ROOT/'tools/contact_coordination/phase_clock_runtime.py'])
    if result.get('loaded_geometry'):key,deps=fingerprint([*[d['path'] for d in deps],result['loaded_geometry']['path'],ROOT/'tools/contact_coordination/loaded_contact_geometry.py'])
    waypoint_record=read(receiver_waypoint/'RESULT.json') if receiver_waypoint else None
    if waypoint_record:
        assert waypoint_record['status']=='VALID' and waypoint_record['connection']==record(connection/'RESULT.json')
        assert waypoint_record['candidate']['candidate_seed']==row['candidate_seed']
        assert waypoint_record['waypoint']==record(receiver_waypoint/'WAYPOINT.npz')
        lead=int(waypoint_record['closing_frames'])
        key,deps=fingerprint([*[d['path'] for d in deps],receiver_waypoint/'RESULT.json',receiver_waypoint/'WAYPOINT.npz'])
    folder=out/'prototype'/sid/'handoff_prefix'/(key[:12]+f'_advance_{lead}'+(f'_receiver_t{receiver_time_scale:g}' if receiver_time_scale!=1 else '')+('_through_dual' if through_dual_only else ''))
    if folder.exists():raise FileExistsError(folder)
    plan=folder/'physical_plan';q=np.load(sub/'PHASE_Q.npz')['q'][:4]
    old=dict(np.load(base/'COMMANDS.npz'));end=np.flatnonzero(old['stage']=='HOLD_ELEVATED')[-1]+1
    commands=list(old['commanded_q_rad'][:end]);labels=list(old['stage'][:end]);intents=list(old['common_task_intent'][:end])
    bounds=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][:14]
    velocity=np.asarray([v['max_velocity_rad_s'] for v in bounds]);acceleration=np.asarray([v['max_acceleration_rad_s2'] for v in bounds])
    fingers=np.asarray(read(out/'target_repair/CONTACT_CALIBRATION.json')['contacts']['pregrasp']['commanded_finger_q']);timing=[]
    def append(a,b,label,intent,minimum):
        points,dt=quintic_retime(np.asarray([a,b]),velocity,acceleration)
        if dt[0]<minimum:
            n=int(np.ceil(minimum*30));u=np.arange(n+1)/n;s=10*u**3-15*u**4+6*u**5;points=a+s[:,None]*(b-a)
        if label=='GIVER_CLEARANCE' and result.get('coordinated_giver_release') and result.get('giver_release_policy')=='middle_first':
            from .phase_clock_runtime import release_arm_fraction
            from tools.common_execution_layer import Dex3Primitive,read_json
            from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
            primitive=Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER))
            n=max(2*primitive.release_frames,len(points)-1+primitive.release_frames)
            points=np.asarray([a+release_arm_fraction(i,n,'middle_first',primitive.release_frames)*(b-a) for i in range(n+1)])
        points=points[1:];commands.extend(np.c_[points,np.tile(fingers,(len(points),1))]);labels.extend([label]*len(points));intents.extend([intent]*len(points))
        timing.append(dict(phase=label,seconds=len(points)/30,frames=len(points)))
    append(q[0],q[1],'LEFT_CARRY','LEFT_HOLD_INTENT',8.)
    append(q[1],q[1],'LEFT_CARRY_STABILIZE','LEFT_HOLD_INTENT',1.)
    if waypoint_record:
        via=np.load(receiver_waypoint/'WAYPOINT.npz')['q']
        append(q[1],via,'RECEIVER_APPROACH','OPEN_INTENT',3.)
        before=len(commands)
        append(via,q[2],'RECEIVER_APPROACH','OPEN_INTENT',lead/30.)
        assert len(commands)-before==lead
    else:append(q[1],q[2],'RECEIVER_APPROACH','OPEN_INTENT',6.*receiver_time_scale)
    boundary=len(commands)
    # Keep the same post-closure observation/release window when some or all
    # of the nominal closing sequence occurs after arm arrival. Otherwise a
    # stationary-close diagnostic ends before release plus retention can occur.
    remaining_close_frames=max(0,qualified_receiver_lead_frames()-lead)
    append(q[2],q[2],'DUAL_SUPPORT','HANDOFF_INTENT',6.5+remaining_close_frames/30.)
    if not through_dual_only:
        append(q[2],q[3],'RIGHT_TRANSPORT' if result.get('receiver_departure') else 'GIVER_CLEARANCE','RIGHT_HOLD_INTENT',6. if result.get('receiver_departure') else 2.)
        append(q[3],q[3],'RIGHT_OWNERSHIP_VERIFY','RIGHT_HOLD_INTENT',2.)
    if lead:
        begin=boundary-lead
        assert labels[begin]=='RECEIVER_APPROACH'
        intents[begin:boundary]=['HANDOFF_INTENT']*lead
    commands=np.asarray(commands)
    coordinated=bool(result.get('coordinated_giver_release',False))
    if through_dual_only and coordinated:raise ValueError('A through-dual control requires stationary release')
    release_boundary=labels.index('GIVER_CLEARANCE') if coordinated else boundary
    atomic_npz(plan/'COMMANDS.npz',commanded_q_rad=commands,raw_policy_command=commands,common_task_intent=np.asarray(intents),stage=np.asarray(labels),common_initial_q_rad=old['common_initial_q_rad'],joint_names=old['joint_names'],control_fps_hz=np.asarray(30.),stable_episode_id=np.asarray(sid),method=np.asarray('b'),qualification_only=np.asarray(True),policy_used=np.asarray(False),giver_release_not_before_frame=np.asarray(release_boundary),giver_release_policy=np.asarray(result.get('giver_release_policy','simultaneous')),receiver_transition_frames=np.asarray(lead),coordinated_giver_release=np.asarray(coordinated),coordinated_release_frames=np.asarray(max(45,labels.count('GIVER_CLEARANCE'))))
    shutil.copy2(base/'SOURCE_SCENE.json',plan/'SOURCE_SCENE.json')
    atomic_json(folder/'SELECTED_CONTACTS.json',contacts)
    atomic_json(plan/'PLAN.json',dict(status='HANDOFF_PREFIX_PENDING_VALIDATION',source_id=sid,full_task=False,diagnostic=True,frames=len(commands),timing=timing,source_conditioned=True,connection=record(connection/'RESULT.json'),dependencies=deps,commands=record(plan/'COMMANDS.npz'),no_placement_candidate_executed=True,through_dual_only=through_dual_only,receiver_time_scale=receiver_time_scale,receiver_advance_frames=lead,receiver_overlap_basis='Receiver arm approach and preshape/close are jointly retimed; same normalized overlap, endpoints and source intent.2x timing is a TRAIN response to measured receiver-acquisition angular-speed validity excursions.' if receiver_time_scale!=1 else 'Qualified preshape + close overlap' if lead else 'Arrive at selected contact pose OPEN, then perform the same qualified preshape/close primitive while stationary. Giver release still requires measured receiver support.'))
    atomic_json(out/'target_repair/CURRENT_HANDOFF_PREFIX.json',dict(path=str(plan.resolve())))
    return plan


if __name__=='__main__':
    import argparse
    from .physical_attempt import validate_full,launch
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--connection',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--capture-observations',action='store_true');p.add_argument('--receiver-advance-frames',type=int);p.add_argument('--receiver-waypoint',type=Path);p.add_argument('--receiver-time-scale',type=float,default=1.);p.add_argument('--through-dual-only',action='store_true');args=p.parse_args()
    plan=build(args.run_dir.resolve(),args.connection.resolve(),args.receiver_advance_frames,args.receiver_waypoint,args.receiver_time_scale,args.through_dual_only);validation=validate_full(args.run_dir.resolve(),plan)
    print(plan,validation['status'],flush=True)
    if args.execute and validation['status']=='VALID':launch(plan,diagnostic=True,capture_observations=args.capture_observations)
