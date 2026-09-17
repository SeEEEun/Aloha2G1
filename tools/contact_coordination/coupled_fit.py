"""Final bounded TRAIN check: pose-region fit with fixed object-in-hand contacts.

Both conditions use the same three seeds, unary factors, 42 joint variables,
workspace, limits and numerical budget. Only the cross-hand residual changes.
Contact transforms are not relaxed. Absolute object pose is a source prior.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .io import ROOT, read, atomic_json, atomic_npz, record, fingerprint
from .targets import se3_error


def fit(g1, source_wrist, relations, seeds, hand_q, region, config, enable_coupling):
    lower=np.tile(g1.arm_limits[:,0]+1e-7,(3,1));upper=np.tile(g1.arm_limits[:,1]-1e-7,(3,1))
    def kinematics(q):
        wrists={s:[] for s in ('left','right')};objects={s:[] for s in wrists}
        for state in q:
            g1.assign(state)
            for side in wrists:
                t=g1.wrist_pose(side);t[:3,3]=g1.model_to_world_position(t[:3,3]);t[:3,:3]=g1.model_to_world_rotation(t[:3,:3])
                wrists[side].append(t);objects[side].append(t@relations[side])
        return {s:np.asarray(v) for s,v in wrists.items()},{s:np.asarray(v) for s,v in objects.items()}
    results=[]
    for seed_index,seed in enumerate(seeds):
        start=time.monotonic();calls=0
        def residual(flat):
            nonlocal calls
            calls+=1;q=flat.reshape(3,14);wrists,objects=kinematics(q);value=[]
            for side in ('left','right'):
                reference=source_wrist[side]
                value.extend((config['unary_position_scale']*(wrists[side][:,:3,3]-reference[:,:3,3])).ravel())
                rotation=wrists[side][:,:3,:3]@reference[:,:3,:3].transpose(0,2,1)
                value.extend((config['unary_rotation_scale']*Rotation.from_matrix(rotation).as_rotvec()).ravel())
                # Conservatively keep the complete oriented proxy above the table.
                half=np.einsum('tij,j->ti',np.abs(objects[side][:,:3,:3]),region['half_extents'])
                position=objects[side][:,:3,3]
                value.extend((config['region_scale']*np.maximum(region['lower']+half-position,0.)).ravel())
                value.extend((config['region_scale']*np.maximum(position+half-region['upper'],0.)).ravel())
            cross_position=objects['left'][:,:3,3]-objects['right'][:,:3,3]
            cross_rotation=Rotation.from_matrix(objects['left'][:,:3,:3]@objects['right'][:,:3,:3].transpose(0,2,1)).as_rotvec()
            factor=1. if enable_coupling else 0.
            value.extend((factor*config['cross_position_scale']*cross_position).ravel())
            value.extend((factor*config['cross_rotation_scale']*cross_rotation).ravel())
            value.extend((config['joint_continuity_scale']*np.diff(q,axis=0)).ravel())
            return np.asarray(value)
        result=least_squares(residual,np.minimum(np.maximum(seed,lower),upper).ravel(),bounds=(lower.ravel(),upper.ravel()),
            max_nfev=config['max_nfev'],ftol=1e-8,xtol=1e-8,gtol=1e-8)
        q=result.x.reshape(3,14)
        # Check a dense joint interpolation across both overlap edges.
        edge=np.vstack([q[0],*[a+np.linspace(0,1,11)[1:,None]*(b-a) for a,b in zip(q[:-1],q[1:])]])
        wrists,objects=kinematics(edge)
        errors=np.asarray([se3_error(a,b) for a,b in zip(objects['left'],objects['right'])])
        geom=g1.trajectory_geometry(edge,np.tile(hand_q['left'],(len(edge),1)),np.tile(hand_q['right'],(len(edge),1)),1e-5)
        workspace_valid=True
        for side in objects:
            half=np.einsum('tij,j->ti',np.abs(objects[side][:,:3,:3]),region['half_extents'])
            position=objects[side][:,:3,3]
            workspace_valid &= bool(np.all(position-half>=region['lower']-1e-6) and np.all(position+half<=region['upper']+1e-6))
        consistent=bool(np.max(errors[:,0])<=config['consistency_position_m'] and np.max(errors[:,1])<=config['consistency_rotation_rad'])
        results.append(dict(seed=seed_index,q=q,overlap_q=edge,predicted_objects=objects,
            cost=float(result.cost),nfev=int(result.nfev),residual_calls=calls,runtime_s=time.monotonic()-start,
            max_overlap_position_disagreement_m=float(errors[:,0].max()),max_overlap_rotation_disagreement_rad=float(errors[:,1].max()),
            overlap_consistent=consistent,workspace_valid=workspace_valid,
            modeled_collision_records=geom['collision_records'],modeled_collision_free=not geom['collision_records'],
            physical_execution_valid=False))
        print('COUPLED_FIT',enable_coupling,seed_index,'disagreement',errors.max(axis=0),'collision_records',len(geom['collision_records']),flush=True)
    return results


def run(out,resume=False):
    from .source_phase import COMMON
    from .prototype import INITIAL
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    source_id=read(out/'bootstrap/SELECTION.json')['prototype_source_id'];phase=read(out/'source_phase'/source_id/'PHASE_RECORD.json')
    priors=dict(np.load(out/'source_phase'/source_id/'SOURCE_PRIORS.npz'))
    frames=np.asarray(phase['handoff_sample_indices']);frames=frames[np.linspace(0,len(frames)-1,3).round().astype(int)]
    source={s:priors[s+'_wrist_world'][frames] for s in ('left','right')}
    relation={s:np.asarray(phase['registered_wrist_object_relations'][s]) for s in source}
    common=load_common_config(COMMON);scene=load_scene(common);g1=G1Kinematics(common,scene)
    natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad'])
    middle=np.mean(g1.arm_limits,axis=1)
    seeds=[np.tile(natural,(3,1)),np.tile(middle,(3,1)),np.tile((natural+middle)/2,(3,1))]
    physics=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')
    names=[x['joint_name'] for x in read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs']]
    hand={}
    for s,i in [('left',14),('right',21)]:
        hand[s]=np.asarray(physics['hand_states'][s]['POWER_GRASP_P14'])[[names[i:i+7].index(n) for n in g1.hand_joint_names[s]]]
    dim=np.asarray(next(x for x in physics['geometry_candidates'] if x['name']=='INTERMEDIATE_PLUSH_PROXY')['dimensions_m'])
    region=dict(lower=np.asarray([0.,0.,scene['table']['surface_height_m']]),
                upper=np.r_[scene['table']['size_xy_m'],scene['table']['surface_height_m']+g1.shoulder_wrist_reach_geometry()['sides']['left']['upper_effective_length_m']+g1.shoulder_wrist_reach_geometry()['sides']['left']['forearm_effective_length_m']],half_extents=dim/2)
    config=dict(max_nfev=120,unary_position_scale=10.,unary_rotation_scale=.1,
                cross_position_scale=100.,cross_rotation_scale=10.,region_scale=100.,joint_continuity_scale=.01,
                consistency_position_m=.003,consistency_rotation_rad=.05)
    folder=out/'prototype'/source_id/'coupled_pose_region_v3'
    atomic_json(folder/'CONFIG.json',dict(numeric=config,region=region,seeds=seeds,
        semantic_permission='No absolute-world doll handoff orientation specified; preserve fixed object-in-hand contact frames and use source world pose as unary prior. This does not qualify contact capability.',
        postfit_runtime_gate='runtime colliders, carried-object geometry and connecting edges not yet certified',
        source_phase=record(out/'source_phase'/source_id/'PHASE_RECORD.json'),implementation=record(__file__),
        status='TRAIN_ONLY_PRE_OUTCOME_BOUNDED_DEVELOPMENT',identical_budget_and_non_coupling_settings=True))
    all_results={}
    for enabled in (True,False):
        results=fit(g1,source,relation,seeds,hand,region,config,enabled)
        for r in results:
            atomic_npz(folder/f'coupling_{enabled}_seed_{r["seed"]}.npz',q=r.pop('q'),overlap_q=r.pop('overlap_q'),
                **{s+'_predicted_object':v for s,v in r.pop('predicted_objects').items()})
        all_results[str(enabled)]=results
        atomic_json(folder/f'coupling_{enabled}.json',results)
    result=dict(status='DEVELOPMENT_DIAGNOSTIC_COMPLETE',results=all_results,
                source_conditioned_full_task='NOT_DEMONSTRATED',physically_run=False,
                attribution='predicted kinematic diagnostic only; no physical coupling attribution')
    atomic_json(folder/'RESULT.json',result)
    return result
