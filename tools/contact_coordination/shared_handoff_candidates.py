"""One source-conditioned endpoint bank; explicit independent/coupled ranking."""
import copy
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json
from .source_phase import COMMON,mean_pose,pose
from .prototype import INITIAL
from .morphology_repair import world_wrist,wrist_target
from .planner import solve_endpoints
from .runtime_hulls import Checker,object_dimensions
from .interaction_candidates import handoff_pair_ranking


def build(out,contact_mode='translation',calibration_path=None,orientation_mode='calibrated_gravity',receiver_contact='receiver_acquisition_intent'):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from .planning_kinematics import G1Kinematics
    from .receiving_relation import build_region,realize_contact
    from .handoff_repair import carry_contact_for_candidate
    from .scientific_cache import key as cache_key
    if contact_mode!='translation':raise ValueError('Repaired shared candidate family supports translation contact chart')
    calibration_path=calibration_path or out/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json'
    cal=read(calibration_path);cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id'];phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
    priors=np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz')
    raw=mean_pose(priors['inferred_object_from_left'][phase['handoff_sample_indices']])
    natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad']);dims=object_dimensions(out)
    contacts=copy.deepcopy({'left':cal['contacts']['giver_handoff_intent'],'right':cal['contacts'][receiver_contact]})
    capture=cal.get('receiver_capture_transition');delta=np.asarray(capture['T_preobject_postobject']) if capture else np.eye(4)
    up=np.asarray(capture['gravity_up_in_preobject']) if capture else np.array([0.,0.,1.])
    separation=np.asarray(read(out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json')['offset_object_frame_m'])
    region=build_region(phase,contacts['right'],dims,separation,up)
    contacts['right']=realize_contact(contacts['right'],region)
    yaw=np.arctan2(raw[1,0],raw[0,0]);source_rotation=Rotation.from_euler('z',yaw).as_matrix()@Rotation.align_vectors([[0.,0.,1.]],[up])[0].as_matrix()
    key=cache_key(out,sid,'receiving_candidate_and_coupled_selection',dict(contact_mode=contact_mode,orientation_mode=orientation_mode,receiver_contact=receiver_contact),[calibration_path])
    folder=out/'prototype'/sid/'shared_handoff_candidates'/key[:12]
    receipt=folder/'CACHE_CONTRACT.json'
    if receipt.exists():
        saved=read(receipt)
        if saved['key']!=key or any(record(a['path'])!=a for a in saved['artifacts']):raise ValueError('Changed shared-bank cache')
        atomic_json(out/'target_repair/CURRENT_CONTACT_REGION_FIT.json',dict(path=str(folder),cache_key=key))
        return {str(flag):read(folder/f'coupling_{flag}.json') for flag in (False,True)}
    ch=Checker(g,out,cal['joint_names']);openhand=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q'])
    projection=[]
    # Independent reachability priors locate the shared-region sampling box.
    # No q_L/q_R consistency objective is optimized by this coarse construction.
    for side in ('left','right'):
        indices=np.arange(7) if side=='left' else np.arange(7,14)
        relation=np.asarray(contacts[side]['T_wrist_H'])@np.asarray(contacts[side]['T_HO'])
        for seed_index,seed in enumerate((natural,np.asarray(contacts[side]['seed_q']))):
            def residual(active):
                q=natural.copy();q[indices]=active;g.assign(q);x=world_wrist(g,side)@relation
                half=np.abs(x[:3,:3])@dims/2
                lo=np.array([0.,0.,.795+.063])+half
                hi=np.array([.835,.72,max(raw[2,3],.795+.063+dims[2])])-half
                return np.r_[x[:3,3]-raw[:3,3],.1*Rotation.from_matrix(source_rotation.T@x[:3,:3]).as_rotvec(),
                    10*(x[:3,:3]@up-[0,0,1]),100*np.maximum(lo-x[:3,3],0),100*np.maximum(x[:3,3]-hi,0),.1*(active-seed[indices])]
            sol=least_squares(residual,np.clip(seed[indices],g.arm_limits[indices,0]+1e-7,g.arm_limits[indices,1]-1e-7),
                bounds=(g.arm_limits[indices,0]+1e-7,g.arm_limits[indices,1]-1e-7),max_nfev=100)
            q=natural.copy();q[indices]=sol.x;g.assign(q)
            projection.append(dict(side=side,seed=seed_index,q=q,object_pose=world_wrist(g,side)@relation,cost=float(sol.cost)))
    # Closest independently realizable projections define a source-conditioned
    # reachable-region prior. A 3x3 object-scale local chart retains alternatives.
    best={s:min((p for p in projection if p['side']==s),key=lambda p:p['cost']) for s in ('left','right')}
    center=np.mean([p['object_pose'][:3,3] for p in best.values()],axis=0)
    # Absolute handoff yaw is unobserved. Sample the independently reachable
    # yaw interval as well as location; freezing source wrist yaw here can make
    # every otherwise reachable contact endpoint impossible for the target.
    gravity_rotation=Rotation.align_vectors([[0.,0.,1.]],[up])[0].as_matrix()
    headings=[float(Rotation.from_matrix(best[s]['object_pose'][:3,:3]@gravity_rotation.T).as_rotvec()[2]) for s in ('left','right')]
    angular_delta=np.arctan2(np.sin(headings[1]-headings[0]),np.cos(headings[1]-headings[0]))
    rotations=[Rotation.from_euler('z',headings[0]+fraction*angular_delta).as_matrix()@gravity_rotation for fraction in (0.,.5,1.)]
    towards=g.root_pose[:3,3]-center;towards[2]=0.;towards/=np.linalg.norm(towards)
    spacing=float(np.min(dims))*.4
    centers=[center,best['left']['object_pose'][:3,3],best['right']['object_pose'][:3,3],center+spacing*towards,center-spacing*towards]
    poses=[pose(rotation,position) for position in centers for rotation in rotations]
    settings=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,
        max_nfev_per_seed_per_goal=100,endpoint_redundancy_families=False,enable_collision_refinement=False)
    def endpoint(side,obj,contact,identifier,offset):
        indices=np.arange(7) if side=='left' else np.arange(7,14);other='right' if side=='left' else 'left'
        target=wrist_target(obj,contact);f=openhand.copy();f[indices]=np.asarray(contact['measured_finger_q'])[indices]
        def unary_geometry(a,q,goal):
            hits=ch.query(np.r_[q,f],obj,(side,),object_environment=True)
            bad=[h for h in hits if not h['allowed_contact'] and not any(b.startswith(other+'_') for b in h['bodies'])]
            return dict(valid=not bad,forbidden_contacts=bad,opposite_arm_deferred_to_common_pair_validation=True)
        goal=dict(name='HANDOFF_'+side,active_hands=[side],wrist_pose_world={side:target},position_tolerance_m=.003,orientation_tolerance_rad=.05)
        solved=solve_endpoints(g,goal,natural,settings,unary_geometry,[best[side]['q'],contact['seed_q']])
        options=solved['phases'][0]['candidates'];valid=[c for c in options if c['admissible']]
        selected=min(valid,key=lambda c:c.get('cost',0.)) if valid else None
        q=np.asarray(selected['q']) if selected else solved['q'][-1];g.assign(q)
        actual=world_wrist(g,side)@np.asarray(contact['T_wrist_H'])@np.asarray(contact['T_HO'])
        source_cost=float(np.sum((actual[:3,3]-raw[:3,3])**2)/.1**2)
        margin=float(np.min(np.minimum(q[indices]-g.arm_limits[indices,0],g.arm_limits[indices,1]-q[indices])))
        posture=float(np.sum((q[indices]-natural[indices])**2)*.01)
        orientation_deviation=np.asarray(contact.get('candidate_local_rotation_vector_rad',[0.,0.,0.]))
        orientation_score=float(orientation_deviation@orientation_deviation/region['closing_line_rotation_bound_rad']**2) if side=='right' else 0.
        row=dict(candidate_id=sid+':HANDOFF:'+identifier,source_id=sid,phase='HANDOFF_'+side.upper(),
            side=side,source_relation=phase['source_functional_tool_object_relations'][side],
            source_object_prior=raw,local_perturbation=dict(object_world_delta=obj[:3,3]-center,contact_local_delta=offset,
                contact_local_rotation_vector_rad=orientation_deviation),
            mode_seed=contact.get('candidate_orientation_mode','independent_endpoint_seeds'),task_space_target=target,object_pose=actual,target_object_pose=obj,
            contact=contact,q=q,IK_result=solved,geometry_result={'valid':bool(valid)},planner_result=None,
            score=dict(source_deviation=source_cost,source_closing_orientation_deviation=orientation_score,posture=posture,joint_margin=margin),
            unary_score=source_cost+orientation_score+posture+.001/(margin+.01),valid=bool(valid))
        return row
    coarse={s:[] for s in ('left','right')}
    for index,obj in enumerate(poses):
        for side in coarse:coarse[side].append(endpoint(side,obj,contacts[side],f'{side}:region{index}:center',np.zeros(3)))
    # Cheap IK/static filters before contact combinations. Unary sum is
    # separable; no cross-hand compatibility enters this shared pruning.
    region_order=sorted(range(len(poses)),key=lambda i:(not(coarse['left'][i]['valid'] and coarse['right'][i]['valid']),
        coarse['left'][i]['unary_score']+coarse['right'][i]['unary_score'],i))[:3]
    bank={'left':[coarse['left'][i] for i in region_order], 'right':[]}
    # Seven orientation modes x three approach-depth samples = the same21
    # endpoints per retained region. Source roll/pitch are unobserved; constraining
    # realization to yaw alone discards legitimate palm/approach freedom. Reuse
    # the existing object-scale angular radius, preserve the source-conditioned
    # contact center and grasp side, and validate the entire closing sweep.
    radius=region['closing_line_rotation_bound_rad']
    modes=[np.zeros(3),*[sign*radius*np.eye(3)[axis] for axis in range(3) for sign in (-1.,1.)]]
    proposals=[dict(offset=separation*f,priority=0,family='contact_depth_and_local_orientation',fraction=f,
        rotation_vector_rad=rotation,orientation_radius_rad=radius)
        for rotation in modes for f in (0.,.5,1.)]
    for index in region_order:
        for ci,proposal in enumerate(proposals):
            contact=copy.deepcopy(contacts['right'])
            t=np.linalg.inv(np.asarray(contact['T_HO']))
            t[:3,:3]=t[:3,:3]@Rotation.from_rotvec(proposal['rotation_vector_rad']).as_matrix()
            contact['T_HO']=np.linalg.inv(t)
            contact['candidate_local_rotation_vector_rad']=proposal['rotation_vector_rad']
            contact['candidate_orientation_mode']='source_prior_local_orientation_mode_'+str(ci//3)
            target=wrist_target(np.eye(4),contact)
            target[:3,3]+=proposal['offset'];contact['T_HO']=np.linalg.inv(target@np.asarray(contact['T_wrist_H']))
            row=endpoint('right',poses[index],contact,f'right:region{index}:contact{ci}',proposal['offset'])
            row['contact_proposal']=dict(proposal,source_closing_rotation_prior_rad=region['closing_line_rotation_rad'],
                existing_orientation_bound_rad=region['closing_line_rotation_bound_rad']);bank['right'].append(row)
    atomic_json(folder/'ENDPOINT_BANK.json',dict(source=record(out/'source_phase'/sid/'PHASE_RECORD.json'),
        raw_prior=raw,independent_reachability_projections=projection,region_center=center,region_poses=poses,
        region_order=region_order,coarse=coarse,bank=bank,methods_share_exact_bank=True,
        fixed_budget=dict(coarse_regions=15,refined_regions=3,contacts_per_region=len(proposals),expensive_pair_validations_per_method=24)))
    atomic_json(folder/'CONFIG.json',dict(contact_mode=contact_mode,orientation_mode=orientation_mode,receiver_contact=receiver_contact,
        calibration=record(calibration_path),schema='shared_endpoint_candidate_bank_v1',source_receiving_region=region))
    results={}
    for enabled in (False,True):
        ranking=handoff_pair_ranking(bank['left'],bank['right'],enabled);rows=[];validations=0
        for rank,pair in enumerate(ranking):
            l=bank['left'][pair['left']];r=bank['right'][pair['right']]
            pair['planner_result']='NOT_EVALUATED';pair['common_physical_validation']='NOT_EVALUATED'
            if not l['valid'] or not r['valid']:
                pair['common_physical_validation']='NO_VALID_CANDIDATE';continue
            # This is the same contact feasibility gate for B/C, after their
            # different rankings. B's scores and chosen unary IDs are unmodified.
            if pair['position_disagreement_m']>.003 or pair['rotation_disagreement_rad']>.05:
                pair['common_physical_validation']='CROSS_HAND_INCOMPATIBLE';continue
            q=natural.copy();q[:7]=l['q'][:7];q[7:]=r['q'][7:]
            cc={'left':l['contact'],'right':r['contact']};xs={'left':l['object_pose'],'right':r['object_pose']}
            overlap=copy.deepcopy(cc)
            for side in overlap:
                overlap[side]['T_HO']=np.asarray(cc[side]['T_HO'])@delta
                if side=='right':overlap[side]['measured_finger_q']=cal['contacts']['handoff_right']['measured_finger_q']
            hand=np.r_[cc['left']['measured_finger_q'][:7],cc['right']['measured_finger_q'][7:]]
            overlap_hand=np.r_[overlap['left']['measured_finger_q'][:7],overlap['right']['measured_finger_q'][7:]]
            hits=ch.query(np.r_[q,hand],xs['left'],('left','right'),object_environment=True)
            hits+=ch.query(np.r_[q,overlap_hand],xs['left']@delta,('left','right'),object_environment=True)
            bad=[h for h in hits if not h['allowed_contact']]
            if not bad:
                from .contact_transition_geometry import check_receiver_closing
                coarse=check_receiver_closing(ch,q,overlap['left']['measured_finger_q'],xs['left']@delta,coarse=True)
                bad+=coarse['forbidden_contacts']
                pair['coarse_contact_rejection']=bool(bad)
            if not bad:
                if validations>=24:
                    pair['common_physical_validation']='FIXED_PAIR_VALIDATION_BUDGET';continue
                validations+=1
                closing=check_receiver_closing(ch,q,overlap['left']['measured_finger_q'],xs['left']@delta)
                bad+=closing['forbidden_contacts']
            pair['common_physical_validation']='COLLISION' if bad else 'VALID'
            rows.append(dict(candidate_id=sid+':PAIR:'+str(rank),contact_candidate=rank,seed=0,q=q,objects=xs,
                contacts=cc,overlap_contacts=overlap,overlap_objects={s:x@delta for s,x in xs.items()},receiver_capture_transition=capture,
                carry_contacts={'left':cal['contacts']['left_carry'],'right':carry_contact_for_candidate(cal['contacts']['right_carry_command_intent'],cc['right'])},
                cost=pair['score'],score=pair,selected_candidate_ids=[l['candidate_id'],r['candidate_id']],
                candidate_priority=0,valid_stationary_overlap=not bad,forbidden=bad,full_path_valid=False,
                position_disagreement_m=pair['position_disagreement_m'],rotation_disagreement_rad=pair['rotation_disagreement_rad']))
        results[str(enabled)]=rows
        atomic_json(folder/f'coupling_{enabled}.json',rows)
        atomic_json(folder/f'RANKING_{enabled}.json',ranking)
    atomic_json(folder/'RESULT.json',dict(status='SHARED_CANDIDATE_SELECTION_COMPLETE',
        valid_counts={k:sum(r['valid_stationary_overlap'] for r in rows) for k,rows in results.items()},
        generated_counts={s:len(v) for s,v in bank.items()},IK_valid_counts={s:sum(v['valid'] for v in bank[s]) for s in bank},
        representation_difference='Independent unary ranking versus explicit shared-object ranking; identical endpoint bank',physical_run=False))
    atomic_json(receipt,dict(key=key,artifacts=[record(p) for p in sorted(folder.glob('*.json'))]))
    atomic_json(out/'target_repair/CURRENT_CONTACT_REGION_FIT.json',dict(path=str(folder),cache_key=key))
    return results
