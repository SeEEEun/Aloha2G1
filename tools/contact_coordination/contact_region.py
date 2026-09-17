"""Bounded geometry-derived handoff relocation with simultaneous admission.

Second TRAIN strategy: keep G_L/G_R and source object orientation; move the
shared object within the table workspace toward the common natural reachable
object center. No demonstrated task configuration is used as a target or seed.
"""
import copy
import numpy as np
from .io import ROOT, read, atomic_json, atomic_npz, record
from .targets import se3_error
from .planner import realize_phase_goals


def investigate(g1, target, source_prior, natural, config, folder):
    base=target['phase_goals'];overlap=[g for g in base if g['name'].startswith('handoff_')]
    relations={s:np.asarray(target['contact_relations'][s]) for s in ('left','right')}
    # Common environment-derived region; source-conditioned contacts preserved.
    g1.assign(natural);nominal=[]
    for side in ('left','right'):
        t=g1.wrist_pose(side);t[:3,3]=g1.model_to_world_position(t[:3,3]);t[:3,:3]=g1.model_to_world_rotation(t[:3,:3])
        nominal.append((t@relations[side])[:3,3])
    center=np.mean(nominal,axis=0)
    scene=read(ROOT/'isaaclab_doll_handoff_scene/scene_layout.json')
    physics=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')
    dimensions=np.asarray(next(x for x in physics['geometry_candidates'] if x['name']=='INTERMEDIATE_PLUSH_PROXY')['dimensions_m'])
    lower=np.r_[dimensions[:2]/2,scene['table']['surface_height_m']+dimensions[2]/2+physics['gates']['minimum_measured_lift_m']]
    upper=np.r_[np.asarray(scene['table']['size_xy_m'])-dimensions[:2]/2,max(x[2] for x in nominal)]
    center=np.minimum(np.maximum(center,lower),upper)
    origin=np.mean([np.asarray(g['predicted_object_pose_world'])[:3,3] for g in overlap],axis=0)
    shifts=np.asarray([fraction*(center-origin) for fraction in (0.,.5,1.)])
    records={};candidate_q={};counter=0
    for side in ('left','right'):
        records[side]=[];candidate_q[side]=[]
        for index,shift in enumerate(shifts):
            goals=copy.deepcopy(overlap)
            for goal in goals:
                goal['active_hands']=[side]
                for hand in ('left','right'):
                    goal['wrist_pose_world'][hand]=np.asarray(goal['wrist_pose_world'][hand]).copy()
                    goal['wrist_pose_world'][hand][:3,3]+=shift
                goal['predicted_object_pose_world']=np.asarray(goal['predicted_object_pose_world']).copy()
                goal['predicted_object_pose_world'][:3,3]+=shift
            ik=realize_phase_goals(g1,goals,natural,config['planner']);q=ik.pop('q')[1:]
            candidate_q[side].append(q)
            unary=0.
            for goal in goals:
                t=goal['source_time_s'];f=int(np.argmin(abs(source_prior['source_timestamp']-t)))
                p,r=se3_error(goal['wrist_pose_world'][side],source_prior[side+'_wrist_world'][f])
                unary+=(p/config['target']['position_scale_m'])**2+(r/config['target']['orientation_scale_rad'])**2
            records[side].append(dict(index=index,translation_world_m=shift,unary_source_cost=unary/len(goals),
                                     goal_satisfied=ik['status']=='PHASE_IK_SATISFIED',ik=ik,goals=goals))
            atomic_json(folder/f'{side}_candidate_{index}.json',records[side][-1])
            atomic_npz(folder/f'{side}_candidate_{index}.npz',q=q)
            print('CONTACT_REGION_CANDIDATE',side,index,records[side][-1]['goal_satisfied'],flush=True)
    physical=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')
    contract_names=[x['joint_name'] for x in read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs']]
    hands={}
    for side,offset in [('left',14),('right',21)]:
        names=contract_names[offset:offset+7]
        q=np.asarray(physical['hand_states'][side]['POWER_GRASP_P14'])
        hands[side]=np.tile(q[[names.index(n) for n in g1.hand_joint_names[side]]],(3,1))
    pairs=[]
    for left in range(3):
        for right in range(3):
            q=np.column_stack((candidate_q['left'][left][:,:7],candidate_q['right'][right][:,7:]))
            geom=g1.trajectory_geometry(q,hands['left'],hands['right'],config['planner']['collision_penetration_tolerance_m'])
            contacts=geom['collision_records']
            predicted={s:[] for s in ('left','right')}
            for state in q:
                g1.assign(state)
                for s in predicted:
                    wrist=g1.wrist_pose(s);wrist[:3,3]=g1.model_to_world_position(wrist[:3,3]);wrist[:3,:3]=g1.model_to_world_rotation(wrist[:3,:3])
                    predicted[s].append(wrist@relations[s])
            error=np.asarray([se3_error(a,b) for a,b in zip(predicted['left'],predicted['right'])])
            cross=float(np.mean((error[:,0]/config['target']['position_scale_m'])**2+(error[:,1]/config['target']['orientation_scale_rad'])**2))
            kinematic=records['left'][left]['goal_satisfied'] and records['right'][right]['goal_satisfied']
            pairs.append(dict(left=left,right=right,kinematic_valid=kinematic,
                modeled_collision_records=contacts,modeled_collision_free=not contacts,
                # Runtime parity and edges remain additional gates; no command export.
                admitted_for_followup=kinematic and not contacts,
                unary_cost=records['left'][left]['unary_source_cost']+records['right'][right]['unary_source_cost'],
                cross_hand_cost=cross,max_predicted_object_position_disagreement_m=float(error[:,0].max()),
                max_predicted_object_rotation_disagreement_rad=float(error[:,1].max())))
            atomic_npz(folder/f'pair_{left}_{right}.npz',q=q,left_predicted_object=np.asarray(predicted['left']),right_predicted_object=np.asarray(predicted['right']))
    eligible=[p for p in pairs if p['admitted_for_followup']]
    selected={}
    for enabled in (False,True):
        selected[str(enabled)]=min(eligible,key=lambda p:(p['unary_cost']+(config['target']['coupling_weight']*p['cross_hand_cost'] if enabled else 0),p['left'],p['right'])) if eligible else None
    result=dict(strategy='GEOMETRY_REGION_WITH_FIXED_SOURCE_CONTACTS',region_lower_world_m=lower,region_upper_world_m=upper,
        nominal_object_center_world_m=center,source_handoff_center_world_m=origin,candidate_shifts_world_m=shifts,
        contact_relations_unchanged=True,source_object_orientation_unchanged=True,
        identical_unary_candidates_kinematics_collision_budget_for_coupling_toggle=True,pairs=pairs,selected=selected,
        executable=False,source_conditioned_full_task='NOT_DEMONSTRATED',
        status='HANDOFF_CANDIDATES_REQUIRE_EDGE_RUNTIME_VALIDATION' if eligible else 'NO_ADMITTED_HANDOFF_IN_BOUNDED_REGION_BANK')
    atomic_json(folder/'RESULT.json',result)
    return result


def run(out,resume=False):
    from .prototype import CONFIG,INITIAL
    from .source_phase import COMMON
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    source_id=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    # Use the unmodified v1 source-derived contact targets; selection has no run-outcome input.
    primary=next(r for r in read(out/'prototype/RESULT.json')['results'] if r['method']=='INTERACTION_OURS')
    target=read(primary['targets']['path']);priors=dict(np.load(out/'source_phase'/source_id/'SOURCE_PRIORS.npz'))
    common=load_common_config(COMMON);g1=G1Kinematics(common,load_scene(common))
    folder=out/'prototype'/source_id/'contact_region_v2'
    atomic_json(folder/'INVOCATION.json',dict(source_target=primary['targets'],implementation=record(__file__),
        fixed_candidate_fractions=[0.,.5,1.],search='six single-hand sequences, nine simultaneous pairs; both toggles reuse exact same solves',
        phase_pose_tolerances_unchanged=True,dev_outcomes_consulted=False))
    return investigate(g1,target,priors,np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad']),read(CONFIG),folder)
