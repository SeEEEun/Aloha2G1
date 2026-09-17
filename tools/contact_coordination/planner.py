"""Small shared SE(3) phase realization adapter over existing G1 FK/SciPy.

It receives constraints, common q0 and budget; no method identifier. Returned
kinematic candidates are not executable until connecting collision checks and
runtime collider agreement are certified.
"""
import time
import os
import itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .targets import se3_error

_INVOCATIONS=itertools.count()


def record_connection(goal,connection):
    """Evidence only: identity never enters the planner's proposals or score."""
    path=os.environ.get('SOURCE_GUIDED_RRT_AUDIT_LOG')
    if not path:return
    import json
    from pathlib import Path
    from .io import default
    prior=goal.get('source_motion_prior') or {};source=prior.get('provenance',{}).get('path')
    value=dict(invocation_id=next(_INVOCATIONS),phase=goal['name'],
        source_id=Path(source).parent.name if source else goal.get('source_id'),
        **{k:connection.get(k) for k in ('algorithm','status','motion_class','rrt_api_called','rrt_search_expanded',
            'source_guided_samples','global_samples','straight_path_selected','nontrivial_rrt_path_selected','selected_path_id','state_checks','edge_checks','budget')})
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('a') as stream:stream.write(json.dumps(value,default=default)+'\n');stream.flush()
    connection['invocation_evidence']=dict(path=str(target),invocation_id=value['invocation_id'])


def assign_kinematic_state(g1,q,calibration=None):
    """Explicit expected loaded FK; returned commands remain unmodified."""
    calibration=calibration or {}
    offset=np.asarray(calibration.get('arm_joint_offset_rad',np.zeros(14)))
    g1.assign(np.asarray(q)+offset)
    extra=calibration.get('uncommanded_joint_positions_rad',{})
    if extra:
        import mujoco
        for name,value in extra.items():
            if name in g1.arm_joint_names:raise ValueError('Uncommanded-state calibration overrides an arm command')
            if hasattr(g1,'named_qpos'):
                if name not in g1.named_qpos:raise ValueError('Unknown calibrated joint: '+name)
                address=g1.named_qpos[name]
            else:
                joint=mujoco.mj_name2id(g1.model,mujoco.mjtObj.mjOBJ_JOINT,name)
                if joint<0:raise ValueError('Unknown calibrated joint: '+name)
                address=g1.model.jnt_qposadr[joint]
            g1.data.qpos[address]=value
        if hasattr(g1,'forward_kinematics'):g1.forward_kinematics()
        else:mujoco.mj_forward(g1.model,g1.data)


def _legacy_endpoint_solver(g1, goals, q0, config, candidate_validator=None, seed_postures=()):
    started = time.monotonic()
    lower, upper = g1.arm_limits[:, 0]+1e-7, g1.arm_limits[:, 1]-1e-7
    previous = np.asarray(q0).copy()
    rows, states = [], [previous.copy()]
    for goal in goals:
        calibration=goal.get('kinematic_calibration',{})
        offset=np.asarray(calibration.get('arm_joint_offset_rad',np.zeros(14)))
        lower=np.maximum(g1.arm_limits[:,0]+1e-7,g1.arm_limits[:,0]+1e-7-offset)
        upper=np.minimum(g1.arm_limits[:,1]-1e-7,g1.arm_limits[:,1]-1e-7-offset)
        for future_offset in goal.get('prospective_arm_joint_offsets_rad',[]):
            future_offset=np.asarray(future_offset)
            lower=np.maximum(lower,g1.arm_limits[:,0]+1e-7-future_offset)
            upper=np.minimum(upper,g1.arm_limits[:,1]-1e-7-future_offset)
        active = goal['active_hands']
        indices = np.concatenate([np.arange(0,7) if s=='left' else np.arange(7,14) for s in active])
        target = goal['wrist_pose_world']
        seeds = [previous, *map(np.asarray, seed_postures), (lower+upper)/2]
        if not seed_postures:
            seeds.insert(1, np.asarray(q0))
        # Fixed redundancy families explore both elbow orientations. These
        # change the IK initialization, never a contact/world target or budget
        # after failure. Identical families are supplied to every method.
        if config.get('endpoint_redundancy_families',True):
            for sign in (-1.,1.):
                for scale in (.6,1.2):
                    posture=previous.copy()
                    for side in active:
                        base=0 if side=='left' else 7
                        posture[base+2]+=sign*scale
                        posture[base+4]-=sign*scale
                    seeds.append(np.clip(posture,lower,upper))
        candidates=[]
        # A jointly selected phase configuration is already a valid IK
        # candidate. Preserve it if its full pose and connecting edge pass;
        # re-solving the same wrist pose can discard contact-relevant posture.
        if 'preferred_q' in goal:
            proposed=previous.copy();proposed[indices]=np.asarray(goal['preferred_q'])[indices]
            if np.all(proposed>=lower) and np.all(proposed<=upper):
                assign_kinematic_state(g1,proposed,calibration);errors={}
                for side in active:
                    actual=g1.wrist_pose(side)
                    actual[:3,3]=g1.model_to_world_position(actual[:3,3]);actual[:3,:3]=g1.model_to_world_rotation(actual[:3,:3])
                    p,r=se3_error(actual,target[side]);errors[side]=dict(position_m=float(p),orientation_rad=float(r))
                satisfied=all(e['position_m']<=goal['position_tolerance_m'] and (goal.get('orientation_region')=='SO3' or e['orientation_rad']<=goal['orientation_tolerance_rad']) for e in errors.values())
                validation=candidate_validator(previous,proposed,goal) if satisfied and candidate_validator else {'valid':satisfied}
                if satisfied and validation['valid']:
                    candidates.append(dict(seed_index=-1,candidate_kind='preferred_phase_configuration',q=proposed,
                        errors=errors,connection_validation=validation,admissible=True,goal_satisfied=True,
                        hard_pose_constraint=True,pose_prior_within_tolerance=True,nfev=0,residual_evaluations=0,cost=0.))
        for seed_index, seed in enumerate(seeds):
            evaluations = 0
            def residual(q_active):
                nonlocal evaluations
                evaluations += 1
                q=previous.copy();q[indices]=q_active
                assign_kinematic_state(g1,q,calibration)
                value=[]
                for side in active:
                    current=g1.wrist_pose(side)
                    position=g1.model_to_world_position(current[:3,3])
                    rotation=g1.model_to_world_rotation(current[:3,:3])
                    reference=np.asarray(target[side])
                    value.extend(config['position_residual_scale']*(position-reference[:3,3]))
                    value.extend(config['orientation_residual_scale']*Rotation.from_matrix(rotation.T@reference[:3,:3]).as_rotvec())
                value.extend(config['joint_prior_scale']*(q_active-previous[indices]))
                return np.asarray(value)
            solution=least_squares(residual,np.minimum(np.maximum(seed[indices],lower[indices]),upper[indices]),
                bounds=(lower[indices],upper[indices]),max_nfev=config['max_nfev_per_seed_per_goal'],
                ftol=1e-10,xtol=1e-10,gtol=1e-10)
            q=previous.copy();q[indices]=solution.x;assign_kinematic_state(g1,q,calibration)
            errors={};realized={}
            for side in active:
                actual=g1.wrist_pose(side)
                actual[:3,3]=g1.model_to_world_position(actual[:3,3])
                actual[:3,:3]=g1.model_to_world_rotation(actual[:3,:3])
                realized[side]=actual.copy()
                p,r=se3_error(actual,target[side]);errors[side]=dict(position_m=float(p),orientation_rad=float(r))
            fidelity=all(e['position_m']<=goal['position_tolerance_m'] and (goal.get('orientation_region')=='SO3' or e['orientation_rad']<=goal['orientation_tolerance_rad']) for e in errors.values())
            # Source priors may be soft objectives. Full SE(3) residuals still
            # enter every optimization and are recorded; collision/path checks
            # and named joint bounds remain mandatory for admission.
            valid=fidelity or goal.get('hard_pose_constraint',True) is False
            validation = candidate_validator(previous,q,goal) if valid and candidate_validator else {'valid':valid}
            candidates.append(dict(seed_index=seed_index,q=q,errors=errors,goal_satisfied=valid,
                hard_pose_constraint=goal.get('hard_pose_constraint',True),pose_prior_within_tolerance=fidelity,
                connection_validation=validation,admissible=valid and validation['valid'],
                nfev=int(solution.nfev),residual_evaluations=evaluations,cost=float(solution.cost),realized_wrist_pose_world=realized))
        if candidate_validator and config.get('enable_collision_refinement',True) and any(c['goal_satisfied'] for c in candidates) and not any(c['admissible'] for c in candidates):
            # One common bounded retry makes collision geometry part of IK,
            # rather than rejecting every exact-pose solution after solving.
            # Contact modes remain in the supplied validator; no method ID or
            # source-specific posture is consumed by this backend.
            def penetration(validation):
                depths={}
                for hit in validation.get('forbidden_contacts',[]):
                    key=tuple(sorted(hit.get('geoms',hit.get('bodies',[str(hit.get('kind'))]))))
                    depths[key]=max(depths.get(key,0.),float(hit.get('depth_m',hit.get('excess_rad',1.))))
                return float(np.linalg.norm(list(depths.values()))) if depths else (0. if validation['valid'] else 1.)
            ranked=[]
            for candidate in candidates:
                if not candidate['goal_satisfied']:continue
                endpoint=candidate_validator(candidate['q'],candidate['q'],goal)
                ranked.append((penetration(endpoint),candidate['cost'],candidate['seed_index'],candidate))
            base=min(ranked,key=lambda row:row[:3])[-1]
            evaluations=0
            def collision_residual(q_active):
                value=residual(q_active)
                candidate=previous.copy();candidate[indices]=q_active
                endpoint=candidate_validator(candidate,candidate,goal)
                return np.r_[value,penetration(endpoint)/1e-5]
            solution=least_squares(collision_residual,base['q'][indices],bounds=(lower[indices],upper[indices]),
                max_nfev=config.get('collision_refinement_max_nfev',config['max_nfev_per_seed_per_goal']),ftol=1e-10,xtol=1e-10,gtol=1e-10)
            q=previous.copy();q[indices]=solution.x;assign_kinematic_state(g1,q,calibration)
            errors={};realized={}
            for side in active:
                actual=g1.wrist_pose(side);actual[:3,3]=g1.model_to_world_position(actual[:3,3]);actual[:3,:3]=g1.model_to_world_rotation(actual[:3,:3])
                realized[side]=actual.copy();p,r=se3_error(actual,target[side]);errors[side]=dict(position_m=float(p),orientation_rad=float(r))
            fidelity=all(e['position_m']<=goal['position_tolerance_m'] and (goal.get('orientation_region')=='SO3' or e['orientation_rad']<=goal['orientation_tolerance_rad']) for e in errors.values())
            valid=fidelity or goal.get('hard_pose_constraint',True) is False
            validation=candidate_validator(previous,q,goal)
            candidates.append(dict(seed_index=len(seeds),candidate_kind='COMMON_COLLISION_REFINEMENT',base_seed=base['seed_index'],q=q,
                errors=errors,goal_satisfied=valid,hard_pose_constraint=goal.get('hard_pose_constraint',True),pose_prior_within_tolerance=fidelity,
                connection_validation=validation,admissible=valid and validation['valid'],nfev=int(solution.nfev),residual_evaluations=evaluations,
                cost=float(np.sum(residual(solution.x)**2)/2),realized_wrist_pose_world=realized,
                collision_depth_normalizer_m=1e-5,additional_seed_count=1))
        selected=min(candidates,key=lambda c:(not c['admissible'],not c['goal_satisfied'],c['cost'],c['seed_index']))
        continuation=None
        if candidate_validator and not selected['admissible'] and goal.get('cartesian_connection_steps',config.get('cartesian_connection_steps',0))>0:
            # Connect the supplied spatial goal with the existing IK backend.
            # A single joint-space chord can sweep through an obstacle even
            # when the task-space approach direction and endpoint are clear.
            from scipy.spatial.transform import Slerp
            assign_kinematic_state(g1,previous,calibration);starts={}
            for side in active:
                t=g1.wrist_pose(side);t[:3,3]=g1.model_to_world_position(t[:3,3]);t[:3,:3]=g1.model_to_world_rotation(t[:3,:3]);starts[side]=t
            count=int(goal.get('cartesian_connection_steps',config.get('cartesian_connection_steps',0)));waypoints=[]
            for step in range(1,count+1):
                u=step/count;poses={}
                for side in active:
                    first=starts[side];last=np.asarray(target[side]);t=first.copy()
                    t[:3,3]=(1-u)*first[:3,3]+u*last[:3,3]
                    t[:3,:3]=Slerp([0.,1.],Rotation.from_matrix([first[:3,:3],last[:3,:3]]))([u]).as_matrix()[0];poses[side]=t
                waypoint=dict(goal,wrist_pose_world=poses,cartesian_connection_steps=0)
                waypoint.pop('preferred_q',None)
                if step<count:waypoint['name']=goal['name']+'_CONNECT'
                waypoints.append(waypoint)
            bounded=dict(config,cartesian_connection_steps=0,max_nfev_per_seed_per_goal=config.get('connection_max_nfev',40))
            continuation=_legacy_endpoint_solver(g1,waypoints,previous,bounded,candidate_validator,seed_postures)
            path=continuation.pop('q')
            if len(path)==count+1 and all(p['admissible'] for p in continuation['phases']):
                final=continuation['phases'][-1]
                chosen=next((c for c in final['candidates'] if c.get('seed_index')==final['selected_seed']),final['candidates'][0])
                selected=dict(chosen,q=path[-1],seed_index='cartesian_continuation',connecting_q=path)
                candidates.append(selected)
        detours=[]
        if candidate_validator and not selected['admissible']:
            for route_index, preposes in enumerate(goal.get('connection_preposes', [])):
                waypoint=dict(goal,name=goal['name']+'_CONNECT',wrist_pose_world=preposes)
                final=dict(goal)
                for item in (waypoint,final):
                    item.pop('connection_preposes',None);item.pop('preferred_q',None)
                bounded=dict(config,max_nfev_per_seed_per_goal=config.get('connection_max_nfev',40))
                route=_legacy_endpoint_solver(g1,[waypoint,final],previous,bounded,candidate_validator,seed_postures)
                route_q=route.pop('q');detours.append(dict(index=route_index,preposes=preposes,result=route))
                if len(route_q)==3 and all(p['admissible'] for p in route['phases']):
                    knots=[previous.copy()]
                    for j,p in enumerate(route['phases']):
                        subpath=p.get('connecting_q')
                        knots.extend(np.asarray(subpath)[1:] if subpath is not None else [route_q[j+1]])
                    last=route['phases'][-1]
                    chosen=next((c for c in last['candidates'] if c.get('seed_index')==last['selected_seed']),last['candidates'][0])
                    selected=dict(chosen,q=route_q[-1],seed_index='geometry_preapproach_'+str(route_index),connecting_q=np.asarray(knots))
                    candidates.append(selected);break
        previous=selected['q'].copy();states.append(previous)
        rows.append(dict(phase=goal['name'],selected_seed=selected['seed_index'],goal_satisfied=selected['goal_satisfied'],admissible=selected['admissible'],
                         hard_pose_constraint=goal.get('hard_pose_constraint',True),pose_prior_within_tolerance=selected['pose_prior_within_tolerance'],
                         selected_errors=selected['errors'],candidates=candidates,
                         connecting_q=selected.get('connecting_q'),cartesian_continuation=continuation,geometry_preapproach=detours))
        if candidate_validator and not selected['admissible']:
            break
    return dict(status='PHASE_IK_SATISFIED' if all(r['goal_satisfied'] for r in rows) else 'PHASE_IK_BUDGET_EXHAUSTED',
                q=np.asarray(states),phases=rows,runtime_s=time.monotonic()-started,
                source_fidelity_is_execution_gate=False,
                command_exported=False,execution_valid=False,
                remaining_validation=['connecting edges','carried-object/runtime colliders','retiming','contact patch support'])


def solve_endpoints(g1, goal, previous, config, endpoint_validator=None, seed_postures=()):
    """IK admission checks only q_goal; this function cannot connect a path."""
    endpoint_goal=dict(goal,cartesian_connection_steps=0,connection_preposes=[])
    bounded=dict(config,cartesian_connection_steps=0)
    check=(lambda a,b,g:endpoint_validator(b,b,g)) if endpoint_validator else None
    result=_legacy_endpoint_solver(g1,[endpoint_goal],previous,bounded,check,seed_postures)
    result['role']='IK_ENDPOINTS_ONLY'
    return result


def realize_phase_goals(g1, goals, q0, config, candidate_validator=None, seed_postures=()):
    """Endpoint IK followed by a real bounded collision-aware path search.

    The old Cartesian-IK continuation and prepose recursion are disabled.
    Contact admission is checked at endpoints; interior validity uses the same
    contact mode with the phase name marked CONNECT (no endpoint-only close).
    """
    from .joint_path_planner import plan, Budget
    started=time.monotonic();states=[np.asarray(q0).copy()];phases=[]
    for goal in goals:
        previous=states[-1]
        if goal.get('source_wrist_reference_waypoints'):
            # A alone supplies this representation. Each geometric keyframe is
            # an endpoint; the common planner still connects all resulting q.
            sequence=[]
            for waypoint in goal['source_wrist_reference_waypoints']:
                item=dict(goal,**waypoint)
                item.pop('source_wrist_reference_waypoints',None)
                item.pop('preferred_q',None)
                # A's successive source wrist targets already define its
                # primary local reference. Repeating the entire phase curve
                # between every pair would introduce artificial loops.
                item.pop('source_motion_prior',None)
                item['name']=goal['name']+'_SOURCE_REFERENCE_CONNECT'
                sequence.append(item)
            final=dict(goal);final.pop('source_wrist_reference_waypoints');final.pop('source_motion_prior',None);sequence.append(final)
            reference=realize_phase_goals(g1,sequence,previous,config,candidate_validator,seed_postures)
            valid=len(reference['phases'])==len(sequence) and all(p['admissible'] for p in reference['phases'])
            knots=[previous]
            for ph in reference['phases']:
                if ph.get('connecting_q') is not None:knots.extend(ph['connecting_q'][1:])
            combined=dict(reference['phases'][-1],phase=goal['name'],admissible=valid,
                connecting_q=np.asarray(knots),source_wrist_reference=reference,
                representation='WRIST_REFERENCE_TRAJECTORY',source_trajectory_waypoints=len(sequence)-1)
            phases.append(combined);states.append(reference['q'][-1])
            if not valid:break
            continue
        endpoints=solve_endpoints(g1,goal,previous,config,candidate_validator,seed_postures)
        phase=endpoints['phases'][0];candidates=phase['candidates']
        ranked=sorted(enumerate(candidates),key=lambda r:(not r[1]['admissible'],r[1].get('cost',0.),r[0]))
        selected=None;attempts=[];connected=[];unique=[]
        active=np.concatenate([np.arange(0,7) if s=='left' else np.arange(7,14) for s in goal['active_hands']])
        for index,candidate in ranked:
            if candidate['admissible'] and not any(np.max(abs(np.asarray(candidate['q'])-np.asarray(v[1]['q'])))<1e-4 for v in unique):unique.append((index,candidate))
        endpoint_budget=int(config.get('connecting_endpoint_budget',2));total_checks=Budget(**config.get('path_planner_budget',{})).state_checks
        for index,candidate in unique[:endpoint_budget]:
            if candidate_validator is None:
                selected=(index,candidate,None);break
            interior=dict(goal,name=goal['name']+'_CONNECT')
            def state(q):return candidate_validator(q,q,interior)
            from dataclasses import replace
            bounds=replace(Budget(**config.get('path_planner_budget',{})),state_checks=max(100,total_checks))
            from .source_motion_prior import MotionGuide
            guide=MotionGuide(g1,goal,previous,candidate['q'],active)
            def clearance(q):
                report=candidate_validator(q,q,dict(interior,_quality_clearance=True))
                return report.get('minimum_clearance_m') if report.get('minimum_clearance_m') is not None else 0.
            if goal.get('motion_class')=='CONSTRAINED_LOCAL_CONTACT_MOTION':
                from .joint_path_planner import Validator
                from .path_quality import evaluate
                checker=Validator(g1.arm_limits[:,0],g1.arm_limits[:,1],state,.005,bounds.state_checks)
                path=np.asarray([previous,candidate['q']]);pos=guide.features(path[0])['wrists'];end=guide.features(path[-1])['wrists']
                distance=float(np.max(np.linalg.norm(end-pos,axis=1)))
                # Only a short ingress is eligible; no large motion exception.
                small=distance<=.025 and np.max(abs(path[-1]-path[0]))<=.35
                direction_ok=True
                for u in np.linspace(0,1,9):
                    actual=guide.features((1-u)*path[0]+u*path[1])['wrists']
                    direction_ok &= bool(np.max(np.linalg.norm(actual-((1-u)*pos+u*end),axis=1))<=.003)
                valid=small and direction_ok and checker.edge(*path)
                connection=dict(status='PATH_FOUND' if valid else 'NO_CONNECTING_PATH',path=path if valid else None,
                    algorithm='CONSTRAINED_LOCAL_CONTACT_MOTION',budget=vars(bounds),motion_class='CONSTRAINED_LOCAL_CONTACT_MOTION',
                    rrt_api_called=False,rrt_search_expanded=0,search_used=False,direct_path_valid=bool(valid),
                    state_checks=checker.calls,edge_checks=checker.edges,ingress_distance_m=distance,approach_direction_valid=direction_ok,
                    full_geometry_revalidated=bool(valid),final_path_quality=evaluate(path,g1.arm_limits[:,0],g1.arm_limits[:,1],active,guide.features,guide.reference,clearance) if valid else None)
            else:
                connection=plan(previous,candidate['q'],g1.arm_limits[:,0],g1.arm_limits[:,1],state,active,bounds,guide,clearance)
                connection['motion_class']='FREE_SPACE'
            total_checks-=connection['state_checks']
            record_connection(goal,connection)
            attempts.append(dict(endpoint_index=index,**connection))
            candidate['planner_result']=connection
            if connection['status']=='PATH_FOUND':connected.append((index,candidate,connection))
            if total_checks<100:break
        if connected:
            selected=min(connected,key=lambda v:(v[2]['final_path_quality']['score'],v[1].get('cost',0.),v[0]))
        if selected is None:
            any_ik=any(c['goal_satisfied'] for c in candidates)
            any_geometry=any(c['admissible'] for c in candidates)
            failure='NO_CONNECTING_PATH' if any_geometry else 'NO_VALID_CANDIDATE' if any_ik else 'NO_IK'
            phase.update(admissible=False,causal_failure=failure,planner_attempts=attempts)
            phases.append(phase);states.append(endpoints['q'][-1]);break
        index,candidate,connection=selected
        phase.update(selected_seed=candidate.get('seed_index','preferred_phase_configuration'),
            selected_errors=candidate['errors'],admissible=True,goal_satisfied=True,
            connecting_q=connection['path'] if connection else None,
            connection_method=('CONSTRAINED_LOCAL_CONTACT_MOTION' if goal.get('motion_class')=='CONSTRAINED_LOCAL_CONTACT_MOTION' else 'COLLISION_AWARE_GLOBAL_PLANNER') if connection else 'STATIC_ENDPOINT',
            selected_endpoint_index=index,selected_path_id=connection.get('selected_path_id','contact_local') if connection else None,
            selected_path_quality=connection.get('final_path_quality') if connection else None,
            endpoint_path_candidates_connected=len(connected),connecting_endpoint_budget=endpoint_budget,
            planner_attempts=attempts,role='IK_ENDPOINT_THEN_PATH_PLANNER')
        states.append(np.asarray(candidate['q']).copy());phases.append(phase)
    return dict(status='PHASE_IK_SATISFIED' if len(phases)==len(goals) and all(p['admissible'] for p in phases) else phases[-1].get('causal_failure','NO_COMPLETE_CHAIN'),
        q=np.asarray(states),phases=phases,runtime_s=time.monotonic()-started,
        command_exported=False,execution_valid=False,remaining_validation=['retiming','physical execution'])


def quintic_retime(knots, velocity, acceleration, fps=30.):
    """Stop-to-stop scalar quintics with analytic per-joint derivative bounds.

    Max |s'|=15/8 and max |s''|=10/sqrt(3). Retiming cannot certify
    collision-free paths, and no observed state is clipped by this function.
    """
    knots=np.asarray(knots);velocity=np.asarray(velocity);acceleration=np.asarray(acceleration)
    if np.any(velocity<=0) or np.any(acceleration<=0):
        raise ValueError('positive named motion bounds required')
    points=[knots[0]];durations=[]
    for a,b in zip(knots[:-1],knots[1:]):
        d=np.abs(b-a)
        duration=max(float(np.max(1.875*d/velocity)),float(np.max(np.sqrt((10/np.sqrt(3))*d/acceleration))),1/fps)
        frames=int(np.ceil(duration*fps));duration=frames/fps
        u=np.arange(1,frames+1)/frames;s=10*u**3-15*u**4+6*u**5
        points.extend(a+s[:,None]*(b-a));durations.append(duration)
    return np.asarray(points),np.asarray(durations)


def realize_goal_region(g1, goals, previous, config, candidate_validator, seed_postures=()):
    """Screen the common region bank before refining the best bounded subset.

    All endpoints/edges retain the supplied full geometry validator. The first
    pass only omits expensive collision optimization of rejected endpoints;
    it does not omit checks from any accepted solution. The same three-seed
    contact/IK policy supplies K=3 for every representation and source.
    """
    # Deduplicate projections before claiming target multiplicity. Distinct IK
    # seeds remain endpoint attempts and are not counted as new task targets.
    unique=[];seen=set()
    for original in goals:
        key=tuple((s,np.asarray(t).round(10).tobytes()) for s,t in sorted(original['wrist_pose_world'].items()))
        if key in seen:continue
        seen.add(key);goal=dict(original)
        goal.setdefault('candidate_id',str(goal.get('source_id','source'))+':'+goal['name']+':region'+str(len(unique)))
        unique.append(goal)
    goals=sorted(unique,key=lambda g:g.get('source_deviation_score',0.))
    generated=[dict(candidate_id=g['candidate_id'],source_id=g.get('source_id'),phase=g['name'],source_relation=g.get('source_relation'),
        task_space_target=g['wrist_pose_world'],local_perturbation=g.get('local_perturbation'),IK_result='NOT_EVALUATED',
        geometry_result='NOT_EVALUATED',planner_result='NOT_EVALUATED',score=g.get('source_deviation_score')) for g in goals]
    attempts=[]
    ranked=[]
    for index,goal in enumerate(goals):
        result=realize_phase_goals(g1,[goal],previous,dict(config,enable_collision_refinement=False),candidate_validator,seed_postures)
        q=result.pop('q');attempts.append(dict(candidate=index,goal=goal,result=result,search_pass='SCREEN'))
        generated[index].update(IK_result=result['phases'][0]['goal_satisfied'],geometry_result=any(c['admissible'] for c in result['phases'][0]['candidates']),planner_result=result['phases'][0]['admissible'])
        if result['phases'][0]['admissible']:
            return dict(valid=True,q=q[-1],goal=goal,selected=index,selected_candidate_id=goal['candidate_id'],generated=generated,attempts=attempts)
        candidates=result['phases'][0]['candidates']
        def rank(candidate):
            validation=candidate['connection_validation']
            depths={}
            for hit in validation.get('forbidden_contacts',[]):
                pair=tuple(sorted(hit.get('geoms',hit.get('bodies',[str(hit.get('kind'))]))))
                depths[pair]=max(depths.get(pair,0.),float(hit.get('depth_m',hit.get('excess_rad',1.))))
            return (not candidate['goal_satisfied'],float(np.linalg.norm(list(depths.values()))) if depths else (0. if validation['valid'] else 1.),candidate['cost'])
        ranked.append((min(map(rank,candidates)),index,goal))
    eligible=[row for row in sorted(ranked,key=lambda row:(row[0],row[1])) if not row[0][0]]
    for _,index,goal in eligible[:config.get('region_refinement_top_k',3)]:
        result=realize_phase_goals(g1,[goal],previous,config,candidate_validator,seed_postures)
        q=result.pop('q');attempts.append(dict(candidate=index,goal=goal,result=result,search_pass='TOP_K_COLLISION_REFINEMENT'))
        if result['phases'][0]['admissible']:
            generated[index].update(IK_result=True,geometry_result=True,planner_result=True)
            return dict(valid=True,q=q[-1],goal=goal,selected=index,selected_candidate_id=goal['candidate_id'],generated=generated,attempts=attempts)
    return dict(valid=False,generated=generated,attempts=attempts)
