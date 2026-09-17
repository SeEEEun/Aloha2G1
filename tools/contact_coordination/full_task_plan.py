"""Connect repaired contact-region goals with the shared bounded IK backend."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json,atomic_npz,fingerprint
from .source_phase import COMMON,pose
from .morphology_repair import world_wrist,wrist_target,assign_named
from .planner import realize_phase_goals,quintic_retime,realize_goal_region
from .runtime_hulls import Checker


def realize_source_region(g,goals,previous,config,validator,seeds=(),*,source_folder):
    """Attach the phase prior after geometry has rebuilt/refined any targets."""
    from .source_motion_prior import attach
    region=[dict(attach(goal,source_folder),motion_class='FREE_SPACE') for goal in goals]
    return realize_goal_region(g,region,previous,config,validator,seeds)


def planar_release_clearance(normals,depths,lower,upper,contact_offset):
    """Small geometric correction inside the already allowed bin region."""
    from scipy.optimize import minimize,LinearConstraint,Bounds
    a=np.asarray(normals)[:,:2];b=np.asarray(depths)+contact_offset
    if np.any(np.linalg.norm(a,axis=1)<1e-8):return None
    sol=minimize(lambda v:float(v@v),np.zeros(2),jac=lambda v:2*v,
                 constraints=[LinearConstraint(a,b,np.inf)],bounds=Bounds(lower,upper),
                 method='SLSQP',options=dict(maxiter=20,ftol=1e-12))
    return sol.x if sol.success and np.all(a@sol.x>=b-1e-9) else None


def delayed_release_goal(checker,q,goal,model,opened,held,primitive):
    """Identify obstructed nonopposing digits from the complete opening sweep."""
    from .loaded_contact_geometry import check as check_loaded,object_pose
    from .phase_clock_runtime import receiver_release_target
    from tools.direct_physical_execution_layer import DIGIT_LOCAL_INDICES
    delayed=set();evidence=[];x=object_pose(checker.g1,q,model)
    for frame in range(primitive.release_frames):
        fingers=np.r_[opened[:7],receiver_release_target(held,opened[7:],frame,0,primitive.release_frames)]
        for hit in check_loaded(checker,np.r_[q,fingers],x,('right',),model):
            if hit['allowed_contact']:continue
            bodies=hit.get('bodies',[])
            digits=[d for d in DIGIT_LOCAL_INDICES if any(b.startswith('right_hand_'+d+'_') for b in bodies)]
            if len(digits)!=1 or not any('TrashBin/' in b for b in bodies):return None
            delayed.add(digits[0]);evidence.append(dict(frame=frame,**hit))
    if not delayed or ('thumb' in delayed and {'index','middle'}&delayed):return None
    return dict(goal,final_release_delayed_digits=sorted(delayed),release_order_evidence=evidence,
        release_order_rule='Open unobstructed digits at placement. An obstructed nonopposing digit may remain flexed during empty-hand retreat and opens at the checked retreat endpoint. Full opening sweep, partial-open retreat edges and final opening are validated. No opposing pair remains closed; no runtime object motion is prescribed.')


def refine_release_goal(checker,q,goal,model,opened,held,primitive,scene):
    """One bounded correction from actual opening-sweep wall normals."""
    from .loaded_contact_geometry import check as check_loaded,object_pose
    from .phase_clock_runtime import receiver_release_target
    x=object_pose(checker.g1,q,model);normals=[];depths=[]
    for frame in range(primitive.release_frames):
        fingers=np.r_[opened[:7],receiver_release_target(held,opened[7:],frame,0,primitive.release_frames,goal.get('final_release_delayed_digits',[]),10*primitive.release_frames)]
        for hit in check_loaded(checker,np.r_[q,fingers],x,('right',),model):
            if hit['allowed_contact']:continue
            bodies=hit.get('bodies',[])
            if len(bodies)!=2 or not any('TrashBin/' in body for body in bodies):return None
            robot=next((i for i,body in enumerate(bodies) if body.startswith('right_')),None)
            if robot is None:return None
            normal=np.asarray(hit['normal_world_geom0_to_geom1'])*(1 if robot==1 else -1)
            normals.append(normal);depths.append(hit['depth_m'])
    if not normals:return None
    prior=np.asarray(goal['object_pose']);center=np.asarray(scene['bin']['center_world_xy_m'])
    half=np.abs(prior[:3,:3])@np.asarray([.1175,.0725,.0775])/2
    margin=np.asarray(scene['bin']['opening_dimensions_xy_m'])/2-half[:2]
    current=prior[:2,3]-center
    offset=float(read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m'])
    correction=planar_release_clearance(normals,depths,-margin-current,margin-current,offset)
    if correction is None:return None
    target=np.asarray(goal['wrist_pose_world']['right']).copy();target[:2,3]+=correction
    xp=prior.copy();xp[:2,3]+=correction
    return dict(goal,object_pose=xp,wrist_pose_world={'right':target},
                release_clearance_refinement=dict(delta_world_xy_m=correction,contact_normals_world=normals,
                    penetration_depths_m=depths,contact_offset_m=offset,max_optimizer_iterations=20,
                    rule='One minimum-norm XY correction from complete actual-collider opening sweep, within unchanged object/bin region. Full IK, all connecting edges and the entire opening sweep are rechecked.'))


def build(out,through_handoff=False,contact_candidate=None,giver_release_policy='simultaneous',coordinated_giver_release=False,receiver_departure=False,loaded_geometry=None,spatial_prior_contract=None,enable_coupling=True):
    if receiver_departure and (coordinated_giver_release or loaded_geometry is None):raise ValueError('Receiver departure follows stationary release and requires explicit loaded geometry calibration')
    loaded_model=read(loaded_geometry) if loaded_geometry else None
    if coordinated_giver_release and giver_release_policy not in ('simultaneous','middle_first'):raise ValueError('Unsupported coordinated release sequence')
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from .planning_kinematics import G1Kinematics
    c=load_common_config(COMMON);scene=load_scene(c);g=G1Kinematics(c,scene);sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    from functools import partial
    realize_region=partial(realize_source_region,source_folder=out/'source_phase'/sid)
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json');names=cal['joint_names'];checker=Checker(g,out,names)
    prior_contract=read(spatial_prior_contract) if spatial_prior_contract else None
    selected_fit=out/'target_repair/CURRENT_CONTACT_REGION_FIT.json'
    if prior_contract:
        bank_path=Path(spatial_prior_contract);candidates=prior_contract.get('candidates',[prior_contract['candidate']])
    else:
        if selected_fit.exists():
            pointer=read(selected_fit);fit_folder=Path(pointer['path']);fit_config=read(fit_folder/'CONFIG.json')
            from .scientific_cache import key as cache_key
            expected=cache_key(out,sid,'receiving_candidate_and_coupled_selection',
                dict(contact_mode=fit_config['contact_mode'],orientation_mode=fit_config['orientation_mode'],receiver_contact=fit_config['receiver_contact']),
                [fit_config['calibration']['path']]+([fit_config['contact_preload_region']['path']] if fit_config['contact_mode']=='thumb_preload' else []))
            if pointer.get('cache_key')!=expected:raise ValueError('Stale selected-contact pointer; regenerate from the current source inputs')
            for artifact in read(fit_folder/'CACHE_CONTRACT.json')['artifacts']:
                if record(artifact['path'])!=artifact:raise ValueError('Changed selected contact bank')
            bank_path=fit_folder/f'coupling_{enable_coupling}.json'
        else:bank_path=sorted((out/'prototype'/sid/'contact_region_morphology').glob(f'*/coupling_{enable_coupling}.json'))[-1]
        bank=read(bank_path)
        candidates=sorted([r for r in bank if r['valid_stationary_overlap']],key=lambda r:(r.get('candidate_priority',0),r['cost'],r['seed']))
    if contact_candidate is not None:candidates=[r for r in candidates if r['contact_candidate']==contact_candidate]
    base=out/'prototype'/sid/'morphology_acquisition_v4';prefix=dict(np.load(base/'COMMANDS.npz'));take=np.flatnonzero(prefix['stage']=='HOLD_ELEVATED')[-1]+1
    source_phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
    initial_x=np.asarray(source_phase['initial_object_pose_world']);openhand=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q']);lefthold=np.asarray(cal['contacts']['pickup']['measured_finger_q'])[:7]
    from .scientific_cache import key as cache_key
    dependency_key=cache_key(out,sid,'phase_ik_connection_retiming',dict(through_handoff=through_handoff,
        contact_candidate=contact_candidate,giver_release_policy=giver_release_policy,
        coordinated_giver_release=coordinated_giver_release,receiver_departure=receiver_departure,
        enable_coupling=enable_coupling),[base/'COMMANDS.npz',bank_path]+([loaded_geometry] if loaded_model else []))
    folder=out/'prototype'/sid/'full_task_connection'/(dependency_key[:12]+('_handoff' if through_handoff else '')+(f'_contact_{contact_candidate}' if contact_candidate is not None else '')+'_'+giver_release_policy+('_coordinated' if coordinated_giver_release else '')+('_receiver_departure' if receiver_departure else ''));all_results=[]
    if (folder/'RESULT.json').exists():
        receipt=read(folder/'CACHE_CONTRACT.json')
        if receipt['key']!=dependency_key:raise ValueError('Connection cache identity mismatch')
        for artifact in receipt['artifacts']:
            if record(artifact['path'])!=artifact:raise ValueError('Changed connection cache result')
        return folder
    if folder.exists():raise FileExistsError('Incomplete immutable connection; preserve and diagnose: '+str(folder))
    config=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=120)
    from .calibration_parameters import scaled_ik_config
    config=scaled_ik_config(out,config,'connection')
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    from .execution_timing import common_primitive
    primitive=common_primitive()
    specs=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][:14]
    velocity=np.asarray([v['max_velocity_rad_s'] for v in specs]);acceleration=np.asarray([v['max_acceleration_rad_s2'] for v in specs])
    complete_candidates=[]
    for candidate in candidates:
        loaded_audits=[]
        x=np.asarray(candidate['objects']['left']);cc=candidate['contacts'];carry=candidate.get('carry_contacts',{'left':cal['contacts']['left_carry'],'right':cal['contacts'].get('right_carry',cc['right'])});righthold=np.asarray(carry['right']['measured_finger_q'])[7:]
        hold_contact=dict(carry['right'])
        if loaded_model:
            loaded_model=dict(loaded_model,measured_T_HO=carry['right']['measured_T_HO'])
            hold_contact['T_HO']=loaded_model['measured_T_HO']
        lefthold=np.asarray(cc['left']['measured_finger_q'])[:7]
        overlap_x=np.asarray(candidate.get('overlap_objects',candidate['objects'])['left']);overlap_cc=candidate.get('overlap_contacts',cc)
        capture=candidate.get('receiver_capture_transition');capture_samples=[]
        if capture:
            d=np.asarray(capture['T_preobject_postobject']);rv=Rotation.from_matrix(d[:3,:3]).as_rotvec()
            capture_samples=[pose(Rotation.from_rotvec(u*rv).as_matrix(),u*d[:3,3]) for u in (.5,1.)]
        qstart=prefix['commanded_q_rad'][take-1,:14];targetq=np.asarray(candidate['q']);g.assign(targetq)
        ltarget=world_wrist(g,'left');rtarget=world_wrist(g,'right')
        from .giver_clearance import goal as open_giver_goal
        after_right=rtarget@np.asarray(carry['right']['T_wrist_H'])@np.asarray(carry['right']['T_HO'])
        physical_rtarget=rtarget
        if loaded_model:
            from .loaded_contact_geometry import object_pose as loaded_object_pose,kinematic_calibration
            after_right=loaded_object_pose(g,targetq,loaded_model);physical_rtarget=world_wrist(g,'right')
        left_clear,clearance_evidence=open_giver_goal(checker,targetq,np.r_[openhand[:7],righthold],after_right,ltarget,float(read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m']))
        # Preserve the source-conditioned selected object's orientation into
        # transport. A ratio of calibration transforms is not a world-frame
        # placement orientation. Geometry-bounded placement tilt is considered
        # below only when the source has no required absolute orientation.
        placement_rotation=after_right[:3,:3].copy()
        bin_x=pose(placement_rotation,np.r_[scene['bin']['center_world_xy_m'],.795+.150+.0775/2+.063]);place_x=bin_x.copy();place_x[2,3]=.795+.150-.010
        target_bin=wrist_target(bin_x,hold_contact);target_place=wrist_target(place_x,hold_contact);retreat=target_place.copy();retreat[2,3]+=.085
        goals=[dict(name='LEFT_CARRY',active_hands=['left'],wrist_pose_world={'left':ltarget},contact_mode='LEFT_HOLD',object_pose=x),
            dict(name='RECEIVER_APPROACH',active_hands=['right'],wrist_pose_world={'right':rtarget},contact_mode='RECEIVE_OPEN',object_pose=x),
            dict(name='GIVER_CLEARANCE',active_hands=['left'],wrist_pose_world={'left':left_clear},contact_mode='DUAL_TO_RIGHT' if coordinated_giver_release else 'RIGHT_HOLD',object_pose=overlap_x),
            dict(name='RIGHT_TRANSPORT',active_hands=['right'],wrist_pose_world={'right':target_bin},contact_mode='RIGHT_HOLD',object_pose=bin_x),
            dict(name='PLACE',active_hands=['right'],wrist_pose_world={'right':target_place},contact_mode='RIGHT_HOLD',object_pose=place_x),
            dict(name='POST_RELEASE_RETREAT',active_hands=['right'],wrist_pose_world={'right':retreat},contact_mode='RELEASE',object_pose=place_x)]
        for goal in goals:goal.update(position_tolerance_m=.003,orientation_tolerance_rad=.05,
            source_id=sid,source_relation=source_phase['source_functional_tool_object_relations'])
        goals[-1].update(cartesian_connection_steps=6,orientation_region='SO3',orientation_region_provenance='After natural release the open receiver has no object/hand orientation requirement. Retraction position, complete hand geometry and the connecting path remain constrained; endpoint orientation is only a prior.')
        goals[2].update(orientation_region='SO3',clearance_geometry=clearance_evidence,orientation_region_provenance='The open giver has no task/object orientation requirement after release. Endpoint orientation is a prior; selected fullSE3 and the entire timed release/withdrawal path remain collision checked. Contact-bearing giver pose before release is unchanged.')
        for goal in goals[:2]:
            goal['preferred_q']=targetq
            goal['cartesian_connection_steps']=6
        if loaded_model:
            for goal in goals[:2]:goal['prospective_arm_joint_offsets_rad']=[loaded_model['arm_measured_minus_command_rad']]
        if prior_contract:
            for goal in goals:
                if goal['name'] in prior_contract['goal_overrides']:
                    goal.update(prior_contract['goal_overrides'][goal['name']]);goal.pop('preferred_q',None)
            tr=np.asarray(goals[4]['wrist_pose_world']['right']).copy();tr[2,3]+=.085
            goals[5]['wrist_pose_world']={'right':tr}
        # A receiver must insert around a supported object. A straight joint
        # or Cartesian chord can sweep its thumb across the giver. Supply a
        # small geometry-derived approach bank to the SAME common connector.
        # It receives poses/constraints only and validates every resulting edge.
        receiver=np.asarray(goals[1]['wrist_pose_world']['right'])
        giver=np.asarray(goals[0]['wrist_pose_world']['left'])
        import mujoco
        shoulder_id=mujoco.mj_name2id(g.model,mujoco.mjtObj.mjOBJ_BODY,'right_shoulder_pitch_link')
        g.assign(targetq)
        toward=g.model_to_world_position(g.data.xpos[shoulder_id])-receiver[:3,3]
        away=receiver[:3,3]-giver[:3,3]
        directions=[v/np.linalg.norm(v) for v in (toward,away,np.array([0.,0.,1.])) if np.linalg.norm(v)>1e-8]
        from .runtime_hulls import object_dimensions
        extent=float(np.min(object_dimensions(out)))
        preposes=[]
        for fraction in (.5,1.):
            for direction in directions:
                t=receiver.copy();t[:3,3]+=fraction*extent*direction;preposes.append({'right':t})
        goals[1].update(connection_preposes=preposes,
            connection_region_provenance='Half/full modeled object short extent toward receiver shoulder, away from giver wrist, or upward. Full pose/hand/object checks and every connecting edge remain required. Same bank construction for all spatial objectives.')
        left_preposes=[]
        shoulder_id=mujoco.mj_name2id(g.model,mujoco.mjtObj.mjOBJ_BODY,'left_shoulder_pitch_link')
        toward=g.model_to_world_position(g.data.xpos[shoulder_id])-giver[:3,3]
        away=giver[:3,3]-receiver[:3,3]
        directions=[v/np.linalg.norm(v) for v in (toward,away,np.array([0.,0.,1.])) if np.linalg.norm(v)>1e-8]
        for fraction in (.5,1.):
            for direction in directions:
                t=giver.copy();t[:3,3]+=fraction*extent*direction;left_preposes.append({'left':t})
        goals[0].update(connection_preposes=left_preposes,
            connection_region_provenance='Same half/full modeled object extent approach bank for giver transport around the passive receiver; left-only object contact is required throughout carry.')
        departure_goals=None
        if receiver_departure:
            from .loaded_contact_geometry import departure_region
            departure_goals=departure_region(checker,targetq,np.r_[openhand[:7],righthold],after_right,physical_rtarget,loaded_model,float(read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m']))
            for goal in departure_goals:goal['kinematic_calibration']=kinematic_calibration(loaded_model)
            goals[2]=departure_goals[0]
        if loaded_model:
            for goal in goals:
                if goal['contact_mode']=='RIGHT_HOLD':goal['kinematic_calibration']=kinematic_calibration(loaded_model)
        if prior_contract is None:
            # Real task-space alternatives exist even if the central target
            # already works. They are constructed before selection; the same
            # selector can reject central ingress and try an interior offset.
            for index in (3,4):
                base_goal=goals[index];obj=np.asarray(base_goal['object_pose'])
                half=(np.abs(obj[:3,:3])@object_dimensions(out)/2)[:2]
                margin=np.maximum(np.asarray(scene['bin']['opening_dimensions_xy_m'])/2-half,0.)
                region=[]
                for offset in (np.zeros(2),np.array([margin[0]/2,0]),np.array([-margin[0]/2,0]),np.array([0,margin[1]/2]),np.array([0,-margin[1]/2])):
                    x_candidate=obj.copy();x_candidate[:2,3]+=offset
                    item=dict(base_goal,object_pose=x_candidate,wrist_pose_world={'right':wrist_target(x_candidate,hold_contact)},
                        candidate_id=sid+':'+base_goal['name']+':'+str(len(region)),local_perturbation=offset,
                        source_deviation_score=float(offset@offset))
                    region.append(item)
                goals[index]['target_region_candidates']=region
        if through_handoff:goals=goals[:3]
        def validate(a,b,goal):
            n=max(2,int(np.ceil(np.max(np.abs(b-a))/.02))+1);bad=[];mode=goal['contact_mode'];clearances=[]
            if mode=='DUAL_TO_RIGHT':
                from .phase_clock_runtime import release_geometry_fingers,release_arm_fraction
                from scipy.spatial.transform import Slerp
                _,duration=quintic_retime(np.asarray([a,b]),velocity,acceleration)
                frames=max(60,int(round(float(duration[0])*30)))
                if giver_release_policy=='middle_first':frames=max(2*primitive.release_frames,frames+primitive.release_frames)
                g.assign(a);end_x=world_wrist(g,'right')@np.asarray(carry['right']['T_wrist_H'])@np.asarray(carry['right']['T_HO'])
                rotations=Slerp([0.,1.],Rotation.from_matrix(np.asarray([overlap_x[:3,:3],end_x[:3,:3]])))
                samples=np.linspace(0,frames,3*frames+1)
                for step in samples:
                    s=release_arm_fraction(step,frames,giver_release_policy,primitive.release_frames);q=a+s*(b-a)
                    f,alpha=release_geometry_fingers(np.asarray(overlap_cc['left']['measured_finger_q'])[:7],np.asarray(overlap_cc['right']['measured_finger_q'])[7:],openhand[:7],righthold,max(0.,step-1),max(frames,primitive.release_frames),giver_release_policy,primitive.release_frames)
                    xp=pose(rotations([alpha]).as_matrix()[0],(1-alpha)*overlap_x[:3,3]+alpha*end_x[:3,3])
                    released=(step>=2*primitive.release_frames) if giver_release_policy=='middle_first' else step>=frames
                    hits=getattr(checker,'query',checker.check)(np.r_[q,f],xp,('right',) if released else ('left','right'))
                    bad.extend([dict(sample=float(step),**v) for v in hits if not v['allowed_contact']])
                return dict(valid=not bad,samples=len(samples),forbidden_count=len(bad),forbidden_contacts=bad[:15],contact_mode=mode,release_frames=max(frames,primitive.release_frames),arm_motion_frames=frames)
            f=openhand.copy()
            if mode in ('LEFT_HOLD','RECEIVE_OPEN'):f[:7]=lefthold
            if mode=='RIGHT_HOLD':f[7:]=righthold
            delayed=goal.get('final_release_delayed_digits',[])
            if mode=='RELEASE' and delayed:
                from .phase_clock_runtime import receiver_release_target
                f[7:]=receiver_release_target(righthold,openhand[7:],primitive.release_frames,0,primitive.release_frames,delayed,10*primitive.release_frames)
            for i,u in enumerate(np.linspace(0,1,n)):
                q=a+u*(b-a);assign_named(g,np.r_[q,f],names)
                side='left' if mode in ('LEFT_HOLD','RECEIVE_OPEN') else 'right'
                predicted=world_wrist(g,side)@np.asarray(carry[side]['T_wrist_H'])@np.asarray(carry[side]['T_HO']) if mode!='RELEASE' else np.asarray(goal['object_pose'])
                if mode=='RIGHT_HOLD' and loaded_model:predicted=loaded_object_pose(g,q,loaded_model)
                allowed=('right',) if mode=='RIGHT_HOLD' else ('left',) if mode=='LEFT_HOLD' else ('left','right')
                if mode=='RIGHT_HOLD' and loaded_model:
                    from .loaded_contact_geometry import check as check_loaded
                    hits=check_loaded(checker,np.r_[q,f],predicted,allowed,loaded_model,audit=loaded_audits)
                else:hits=getattr(checker,'query',checker.check)(np.r_[q,f],predicted,allowed,object_environment=mode in ('LEFT_HOLD','RECEIVE_OPEN'))
                bad.extend([dict(sample=i,**v) for v in hits if not v['allowed_contact']])
                if goal.get('_quality_clearance'):
                    if mode=='RIGHT_HOLD' and loaded_model:
                        from .loaded_contact_geometry import predicted_q
                        clearances.append(checker.clearance(predicted_q(np.r_[q,f],loaded_model),predicted,allowed,
                            all_joint_state=loaded_model.get('uncommanded_joint_positions_rad'),object_environment=True))
                    else:clearances.append(checker.clearance(np.r_[q,f],predicted,allowed,object_environment=mode in ('LEFT_HOLD','RECEIVE_OPEN')))
            if mode=='RECEIVE_OPEN' and not goal['name'].endswith('_CONNECT'):
                # The current controller closes only after the receiver has
                # arrived. Future capture deflection is not an earlier object
                # pose along the open-hand approach. Validate that transition
                # at this endpoint for every supplied spatial objective.
                assign_named(g,np.r_[b,f],names)
                incoming=world_wrist(g,'left')@np.asarray(carry['left']['T_wrist_H'])@np.asarray(carry['left']['T_HO'])
                for fraction,relations in [(0.,cc),(1.,overlap_cc)]:
                    transition=incoming@(np.asarray(capture['T_preobject_postobject']) if capture and fraction else np.eye(4))
                    closed=np.r_[relations['left']['measured_finger_q'][:7],relations['right']['measured_finger_q'][7:]]
                    for hit in getattr(checker,'query',checker.check)(np.r_[b,closed],transition,('left','right')):
                        if not hit['allowed_contact']:bad.append(dict(receiver_endpoint_capture_fraction=fraction,**hit))
                from .contact_transition_geometry import check_receiver_closing
                closing=check_receiver_closing(checker,b,overlap_cc['left']['measured_finger_q'],
                    incoming@(np.asarray(capture['T_preobject_postobject']) if capture else np.eye(4)))
                bad.extend(closing['forbidden_contacts'])
            if goal['name'] in ('PLACE','POST_RELEASE_RETREAT'):
                # Placement must admit the actual next hand-opening action,
                # not just the closed-hand endpoint above/inside the bin.
                from .phase_clock_runtime import receiver_release_target
                for step in range(primitive.release_frames):
                    opening=np.r_[openhand[:7],receiver_release_target(righthold,openhand[7:],step if goal['name']=='PLACE' else step+primitive.release_frames,0,primitive.release_frames,delayed,10*primitive.release_frames if goal['name']=='PLACE' else primitive.release_frames)]
                    if loaded_model and goal['name']=='PLACE':
                        xp=loaded_object_pose(g,b,loaded_model)
                        hits=check_loaded(checker,np.r_[b,opening],xp,('right',),loaded_model,audit=loaded_audits)
                    else:
                        g.assign(b);xp=np.asarray(goal['object_pose']) if mode=='RELEASE' else world_wrist(g,'right')@np.asarray(carry['right']['T_wrist_H'])@np.asarray(carry['right']['T_HO'])
                        hits=getattr(checker,'query',checker.check)(np.r_[b,opening],xp,('right',))
                    bad.extend([dict(release_sample=step,**v) for v in hits if not v['allowed_contact']])
            return dict(valid=not bad,samples=n,forbidden_count=len(bad),forbidden_contacts=bad[:15],contact_mode=mode,minimum_clearance_m=min(clearances) if clearances else None)
        def connect(sequence,start,settings,candidate_validator=validate,seed_postures=()):
            states=[np.asarray(start).copy()];phases=[];elapsed=0.;last={}
            for item in sequence:
                from .source_motion_prior import attach
                guided=attach(item,out/'source_phase'/sid);item.update(guided);item['motion_class']='FREE_SPACE'
                if item.get('target_region_candidates'):
                    item['target_region_candidates']=[attach(c,out/'source_phase'/sid) for c in item['target_region_candidates']]
                if item['contact_mode']=='RELEASE':
                    if loaded_model:item['object_pose']=loaded_object_pose(g,states[-1],loaded_model)
                    else:
                        g.assign(states[-1]);item['object_pose']=world_wrist(g,'right')@np.asarray(carry['right']['T_wrist_H'])@np.asarray(carry['right']['T_HO'])
                if item.get('target_region_candidates'):
                    import copy
                    selection=realize_region(g,item['target_region_candidates'],states[-1],settings,candidate_validator,seed_postures)
                    last=copy.deepcopy(selection['attempts'][-1]['result'])
                    last=dict(last,q=np.asarray([states[-1],selection['q'] if selection['valid'] else last['phases'][-1]['candidates'][0]['q']]))
                    last['phases'][0]['target_region_selection']=selection
                    if selection['valid']:
                        item.update(selection['goal'])
                else:last=realize_phase_goals(g,[item],states[-1],settings,candidate_validator,seed_postures)
                if item['contact_mode']=='LEFT_HOLD' and not last['phases'][0]['admissible']:
                    from .passive_hand_clearance import connect as prepare_passive_hand
                    repaired,preparations=prepare_passive_hand(g,item,states[-1],settings,
                        candidate_validator,seed_postures,'right',float(np.min(object_dimensions(out))))
                    if repaired is not None:
                        repaired['phases'][0]['unprepared_connection_failure']=last['phases'][0]
                        last=repaired
                    else:last['phases'][0]['passive_hand_preparation_failures']=preparations
                path=last.pop('q');states.append(path[-1]);phases.extend(last['phases']);elapsed+=last['runtime_s']
                atomic_json(folder/'PHASE_PROGRESS.json',dict(contact_candidate=candidate['contact_candidate'],seed=candidate['seed'],
                    phase=item['name'],admissible=last['phases'][-1]['admissible'],elapsed_s=elapsed,
                    selected_errors=last['phases'][-1]['selected_errors'],checker_statistics=checker.query_statistics))
                if not last['phases'][-1]['admissible']:break
            return dict(last,q=np.asarray(states),phases=phases,runtime_s=elapsed)
        departure_trials=None
        if receiver_departure:
            r=connect(goals[:2],qstart,config,candidate_validator=validate,seed_postures=[targetq]);q=r.pop('q')
            if len(q)==3 and all(p['admissible'] for p in r['phases']):
                if prior_contract:
                    # The shared safety connection starts from realized wrists;
                    # it does not recouple the independent spatial objectives.
                    g.assign(q[2]);candidate['q']=q[2].copy()
                    candidate['objects']={side:world_wrist(g,side)@np.asarray(cc[side]['T_wrist_H'])@np.asarray(cc[side]['T_HO']) for side in cc}
                    candidate['overlap_objects']={side:world_wrist(g,side)@np.asarray(overlap_cc[side]['T_wrist_H'])@np.asarray(overlap_cc[side]['T_HO']) for side in cc}
                    overlap_x=np.asarray(candidate['overlap_objects']['left'])
                    actual_x=loaded_object_pose(g,q[2],loaded_model);actual_wrist=world_wrist(g,'right')
                    departure_goals=departure_region(checker,q[2],np.r_[openhand[:7],righthold],actual_x,actual_wrist,loaded_model,float(read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m']))
                    for goal in departure_goals:goal['kinematic_calibration']=kinematic_calibration(loaded_model)
                for goal in departure_goals:
                    goal.update(source_id=sid,source_relation=source_phase['source_functional_tool_object_relations'],
                        local_perturbation=goal.get('clearance_geometry'),mode_seed='receiver_departure_geometry_family')
                departure_trials=realize_region(g,departure_goals,q[-1],config,validate,[targetq])
                r['phases'].append(departure_trials['attempts'][-1]['result']['phases'][0])
                if departure_trials['valid']:
                    q=np.vstack([q,departure_trials['q']]);goals[2]=departure_trials['goal']
                    if len(goals)>3:
                        tail=connect(goals[3:],q[-1],config,validate,[targetq]);qt=tail.pop('q');q=np.vstack([q,qt[1:]]);r['phases']+=tail['phases']
                departure_trials.pop('q',None)
        else:
            r=connect(goals,qstart,config,candidate_validator=validate,seed_postures=[targetq]);q=r.pop('q')
        transport_trials=None
        if len(goals)>3 and goals[3].get('object_region_enabled',True) and len(q)==5 and all(p['admissible'] for p in r['phases'][:3]) and not r['phases'][3]['admissible']:
            # Preserve the verified carried object's gravity direction. Only
            # world yaw is free when source placement orientation is UNKNOWN;
            # a local-object yaw would change gravity loading of a tilted grasp.
            region=[]
            orientation_height_bank=[(yaw,0.,height) for height in (0.,1.) for yaw in (0.,-np.pi/2,np.pi/2,np.pi)]
            for yaw,pitch,height_fraction in orientation_height_bank:
                xp=bin_x.copy();xp[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@placement_rotation
                half=np.abs(xp[:3,:3])@np.asarray([.1175,.0725,.0775])/2
                xp[2,3]=.795+.150+.063+half[2]
                xp[2,3]+=height_fraction*max(0.,x[2,3]-xp[2,3])
                if np.any(half[:2]>np.asarray(scene['bin']['opening_dimensions_xy_m'])/2):continue
                gp=dict(goals[3]);gp['object_pose']=xp;gp['wrist_pose_world']={'right':wrist_target(xp,hold_contact)}
                gp['orientation_permission']='Absolute source placement orientation is UNKNOWN. Preserve the verified object gravity direction in hand; permit only world-yaw quarter turns whose entire projected object fits the existing bin opening. FullSE3 endpoint and carried geometry on every edge are required. Two geometry/previous-handoff-derived approach heights; no source/DEV outcome-dependent world waypoint.'
                region.append(gp)
            transport_trials=realize_region(g,region,q[3],config,validate,[targetq])
            if transport_trials['valid']:
                q=np.vstack([q[:4],transport_trials['q']]);goals[3]=transport_trials['goal'];r['phases'][3]=transport_trials['attempts'][-1]['result']['phases'][0]
                placement_rotation=np.asarray(goals[3]['object_pose'])[:3,:3]
                place_x[:3,:3]=placement_rotation;goals[4]['object_pose']=place_x;goals[4]['wrist_pose_world']={'right':wrist_target(place_x,hold_contact)}
                tr=goals[4]['wrist_pose_world']['right'].copy();tr[2,3]+=.085;goals[5]['object_pose']=place_x;goals[5]['wrist_pose_world']={'right':tr}
                tail=connect(goals[4:],q[-1],config,validate,[targetq]);qt=tail.pop('q');q=np.vstack([q,qt[1:]]);r['phases']+=tail['phases']
            transport_trials.pop('q',None)
        placement_trials=None
        if len(goals)>4 and goals[4].get('object_region_enabled',True) and len(q)==6 and all(p['admissible'] for p in r['phases'][:4]) and not r['phases'][4]['admissible']:
            # The bin specifies an interior region, not equality to its center.
            # Use conservative full proxy XY extents and the existing opening.
            region=[]
            for pitch in (0.,-np.pi/4,np.pi/4):
                rotation=placement_rotation@Rotation.from_euler('y',pitch).as_matrix()
                half=np.abs(rotation)@np.asarray([.1175,.0725,.0775])/2
                margin=np.asarray(scene['bin']['opening_dimensions_xy_m'])/2-half[:2]
                if np.any(margin<0):continue
                offsets=sorted([np.array([a*margin[0],b*margin[1]]) for a in (-1,0,1) for b in (-1,0,1)],key=lambda v:(float(v@v),float(v[0]),float(v[1])))
                # The authoritative valid bin interval is below the rim, not
                # equality to an arbitrary 10 mm insertion depth. Include its
                # upper interior point inset by the existing contact offset.
                contact_offset=float(read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m'])
                # Source evidence specifies disposal into the bin, not a
                # compulsory COM insertion depth at finger opening. Include a
                # low gravity release with the object's bottom at the rim.
                # This uses the actual object extent and bin geometry; final
                # measured bin entry/settling remains the unchanged task gate.
                upper_z=.795+.150+half[2]+contact_offset
                for delta,z_goal in [(delta,z) for z in (place_x[2,3],upper_z) for delta in offsets]:
                    xp=place_x.copy();xp[:2,3]+=delta;xp[2,3]=z_goal;xp[:3,:3]=rotation
                    half=np.abs(xp[:3,:3])@np.asarray([.1175,.0725,.0775])/2
                    if np.any(np.abs(delta)+half[:2]>np.asarray(scene['bin']['opening_dimensions_xy_m'])/2+1e-9):continue
                    gp=dict(goals[4]);gp['object_pose']=xp;gp['wrist_pose_world']={'right':wrist_target(xp,hold_contact)}
                    gp['orientation_permission']='Source release height and absolute object orientation are unobserved. Preserve grasp and full projected bin-opening fit; permit upright/+/-45 degree pitch and two source-task-consistent release heights: interior COM or object bottom at rim plus contact offset. Both require actual hand opening and carried-object path geometry. Measured bin entry and settling, not endpoint labels, establish placement.'
                    region.append(gp)
            placement_trials=realize_region(g,region,q[4],config,validate,[targetq])
            if not placement_trials['valid'] and loaded_model:
                refinements=[]
                for prior_trial in placement_trials['attempts']:
                    ph=prior_trial['result']['phases'][0]
                    chosen=next(c for c in ph['candidates'] if c.get('seed_index')==ph['selected_seed'])
                    bad=chosen['connection_validation'].get('forbidden_contacts',[])
                    if not ph['goal_satisfied'] or not bad or not all('release_sample' in h for h in bad):continue
                    refined=refine_release_goal(checker,np.asarray(chosen['q']),prior_trial['goal'],loaded_model,openhand,righthold,primitive,scene)
                    if refined is not None:refinements.append(refined)
                    if len(refinements)>=3:break
                if refinements:
                    adjusted=realize_region(g,refinements,q[4],config,validate,[targetq])
                    adjusted['unrefined_region_attempts']=placement_trials['attempts'];placement_trials=adjusted
            if not placement_trials['valid'] and loaded_model:
                delayed_goals=[]
                for prior_trial in placement_trials.get('unrefined_region_attempts',placement_trials['attempts']):
                    ph=prior_trial['result']['phases'][0]
                    chosen=next(c for c in ph['candidates'] if c.get('seed_index')==ph['selected_seed'])
                    bad=chosen['connection_validation'].get('forbidden_contacts',[])
                    if not ph['goal_satisfied'] or not bad or not all('release_sample' in h for h in bad):continue
                    alternative=delayed_release_goal(checker,np.asarray(chosen['q']),prior_trial['goal'],loaded_model,openhand,righthold,primitive)
                    if alternative is not None:delayed_goals.append(alternative)
                    if len(delayed_goals)>=3:break
                if delayed_goals:
                    adjusted=realize_region(g,delayed_goals,q[4],config,validate,[targetq])
                    adjusted['simultaneous_release_attempts']=placement_trials;placement_trials=adjusted
            if placement_trials['valid']:
                # Minimize gravity fall within the SAME bin volume, after a
                # contact-compatible placement orientation/XY region is found.
                # Three lower heights are fixed geometric fractions; no contact
                # tolerance or simulator setting is altered to admit an impact.
                incumbent=placement_trials['goal'];xp=np.asarray(incumbent['object_pose'])
                half_z=float((np.abs(xp[:3,:3])@np.asarray([.1175,.0725,.0775])/2)[2])
                bin_meta=read(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json')['bin_geometry']
                offset=float(read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m'])
                floor_z=.795+float(bin_meta['bottom_thickness_m'])
                lowest=floor_z+half_z+offset;lower_region=[]
                center=np.asarray(scene['bin']['center_world_xy_m']);opening=np.asarray(scene['bin']['opening_dimensions_xy_m'])/2
                old_margin=opening-(np.abs(xp[:3,:3])@np.asarray([.1175,.0725,.0775])/2)[:2]
                normalized_offset=np.divide(xp[:2,3]-center,old_margin,out=np.zeros(2),where=old_margin>1e-9)
                for yaw in (0.,-np.pi/2,np.pi/2,np.pi):
                    rotation=Rotation.from_euler('z',yaw).as_matrix()@xp[:3,:3]
                    half=np.abs(rotation)@np.asarray([.1175,.0725,.0775])/2;margin=opening-half[:2]
                    if np.any(margin<0):continue
                    offsets=[normalized_offset*margin] if yaw==0 else [np.zeros(2),normalized_offset*margin]
                    for fraction in (0.,.5,.75):
                        z=lowest+fraction*(xp[2,3]-lowest)
                        if z>=xp[2,3]-1e-6:continue
                        for delta in offsets:
                            obj=xp.copy();obj[:3,:3]=rotation;obj[:2,3]=center+delta;obj[2,3]=z
                            lower_region.append(dict(incumbent,object_pose=obj,wrist_pose_world={'right':wrist_target(obj,hold_contact)},
                                release_height_region=dict(bin_floor_world_z_m=floor_z,object_projected_half_z_m=half_z,contact_offset_m=offset,
                                    world_yaw_rotation_rad=yaw,fraction_above_lowest=fraction,original_COM_z_m=float(xp[2,3]),rule='Three fixed heights from proxy bottom+floor+contact offset toward admissible insertion height; quarter-turn world yaw preserves grasp gravity direction when source absolute orientation is UNKNOWN. Require full projected object fit; use bin center and the same normalized interior-boundary offset. Existing fullSE3, entire path and opening geometry checks remain mandatory.')))
                if lower_region:
                    lower_region.sort(key=lambda goal:float(np.asarray(goal['object_pose'])[2,3]))
                    lowered=realize_region(g,lower_region,q[4],config,validate,[targetq])
                    if not lowered['valid'] and loaded_model:
                        corrections=[]
                        for trial in lowered['attempts']:
                            ph=trial['result']['phases'][0]
                            chosen=next(c for c in ph['candidates'] if c.get('seed_index')==ph['selected_seed'])
                            if not ph['goal_satisfied']:continue
                            refined=refine_release_goal(checker,np.asarray(chosen['q']),trial['goal'],loaded_model,openhand,righthold,primitive,scene)
                            if refined is not None:corrections.append(refined)
                            if len(corrections)>=3:break
                        if corrections:
                            adjusted=realize_region(g,corrections,q[4],config,validate,[targetq])
                            adjusted['uncorrected_height_attempts']=lowered;lowered=adjusted
                    if lowered['valid']:
                        lowered['higher_release_region']=placement_trials;placement_trials=lowered
                    else:placement_trials['lower_release_search']=lowered
                q=np.vstack([q[:5],placement_trials['q']]);goals[4]=placement_trials['goal'];r['phases'][4]=placement_trials['attempts'][-1]['result']['phases'][0]
                tr=goals[4]['wrist_pose_world']['right'].copy();tr[2,3]+=.085
                goals[5]['wrist_pose_world']={'right':tr};goals[5]['object_pose']=goals[4]['object_pose']
                goals[5]['final_release_delayed_digits']=goals[4].get('final_release_delayed_digits',[])
                tail=connect([goals[5]],q[-1],config,validate,[targetq]);qt=tail.pop('q');q=np.vstack([q,qt[1:]]);r['phases']+=tail['phases']
            placement_trials.pop('q',None)
        # Stationary dual overlap and giver opening are checked over the full
        # calibrated finger interval, rather than one endpoint finger posture.
        overlap=[]
        if len(q)>=3:
            assign_named(g,np.r_[q[2],openhand],names)
            after_release=world_wrist(g,'right')@np.asarray(carry['right']['T_wrist_H'])@np.asarray(carry['right']['T_HO'])
            if loaded_model:after_release=loaded_object_pose(g,q[2],loaded_model)
            from scipy.spatial.transform import Slerp
            rotation_path=Slerp([0.,1.],Rotation.from_matrix(np.asarray([overlap_x[:3,:3],after_release[:3,:3]])))
            release_frames=primitive.release_frames*(2 if giver_release_policy!='simultaneous' else 1)
            # With coordinated withdrawal, the opening sweep is validated on
            # the moving GIVER_CLEARANCE edge above. Only stationary dual support
            # belongs at this unchanged arm configuration.
            for u in ([0.] if coordinated_giver_release else np.linspace(0,1,release_frames*3+1)):
                from .phase_clock_runtime import release_geometry_fingers
                f,alpha=release_geometry_fingers(np.asarray(overlap_cc['left']['measured_finger_q'])[:7],np.asarray(overlap_cc['right']['measured_finger_q'])[7:],openhand[:7],righthold,u*(release_frames-1),release_frames,giver_release_policy,primitive.release_frames)
                xp=pose(rotation_path([alpha]).as_matrix()[0],(1-alpha)*overlap_x[:3,3]+alpha*after_release[:3,3])
                if loaded_model:
                    from .loaded_contact_geometry import check as check_loaded
                    hits=check_loaded(checker,np.r_[q[2],f],xp,('left','right'),loaded_model,alpha=alpha,verify_relation=False,audit=loaded_audits)
                else:hits=getattr(checker,'query',checker.check)(np.r_[q[2],f],xp,('left','right'))
                overlap.extend([dict(open_fraction=float(u),**v) for v in hits if not v['allowed_contact']])
        r['giver_opening_forbidden']=overlap;r['candidate_seed']=candidate['seed'];r['contact_candidate']=candidate['contact_candidate'];r['placement_region_search']=placement_trials;r['transport_region_search']=transport_trials;r['receiver_departure_search']=departure_trials
        if loaded_model:r['loaded_geometry_predictions']=loaded_audits
        sub=folder/f"contact_{candidate['contact_candidate']}_seed_{candidate['seed']}";atomic_json(sub/'PHASE_IK.json',r);atomic_npz(sub/'PHASE_Q.npz',q=q);atomic_json(sub/'GOALS.json',goals)
        all_results.append(dict(candidate_seed=candidate['seed'],contact_candidate=candidate['contact_candidate'],subdirectory=sub.name,completed_phases=len(q)-1,phase_count=len(goals),opening_collisions=len(overlap),all_admissible=all(p['admissible'] for p in r['phases']) and len(q)==len(goals)+1 and not overlap))
        print('FULL_CONNECTION',all_results[-1],flush=True)
        if all_results[-1]['all_admissible']:
            candidate=dict(candidate,full_path_valid=True)
            if candidate.get('score'):candidate['score']=dict(candidate['score'],planner_result='COMPLETE_CHAIN_CONNECTED')
            path_scores=[p['selected_path_quality']['score'] for p in r['phases'] if p.get('selected_path_quality')]
            representation_cost=float(candidate['cost']);quality=.3*representation_cost/(1.+representation_cost)+.7*float(np.mean(path_scores))
            all_results[-1]['complete_chain_quality']=quality;all_results[-1]['mean_path_quality']=float(np.mean(path_scores))
            complete_candidates.append((quality,len(all_results)-1,candidate))
            if len(complete_candidates)>=2:break
    if complete_candidates:
        quality,index,candidate=min(complete_candidates,key=lambda v:(v[0],v[1]))
        # Consumers use the first admissible result; put the actual quality
        # selection first while preserving every evaluated chain record.
        chosen=all_results.pop(index);all_results.insert(0,chosen)
        candidate['complete_chain_quality']=quality
        atomic_json(folder/'SELECTED_CONTACTS.json',candidate)
    atomic_json(folder/'RESULT.json',dict(status=('HANDOFF_PREFIX_BUILT' if through_handoff else 'FULL_PATH_BUILT') if any(r['all_admissible'] for r in all_results) else 'NO_COMPLETE_PATH_WITHIN_BUDGET',results=all_results,contact_bank=record(bank_path),enable_coupling=None if prior_contract else enable_coupling,spatial_prior_contract=record(spatial_prior_contract) if prior_contract else None,implementation=record(__file__),through_handoff_only=through_handoff,giver_release_policy=giver_release_policy,coordinated_giver_release=coordinated_giver_release,receiver_departure=receiver_departure,loaded_geometry=record(loaded_geometry) if loaded_model else None,physical_run=False))
    atomic_json(folder/'CACHE_CONTRACT.json',dict(key=dependency_key,stage='phase_ik_connection_retiming',
        artifacts=[record(p) for p in sorted(folder.rglob('*')) if p.is_file()]))
    return folder


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--through-handoff',action='store_true');p.add_argument('--contact-candidate',type=int);p.add_argument('--giver-release-policy',choices=['simultaneous','thumb_first','middle_first'],default='simultaneous');p.add_argument('--coordinated-giver-release',action='store_true');p.add_argument('--receiver-departure',action='store_true');p.add_argument('--loaded-geometry',type=Path);a=p.parse_args();print(build(a.run_dir,a.through_handoff,a.contact_candidate,a.giver_release_policy,a.coordinated_giver_release,a.receiver_departure,a.loaded_geometry))
