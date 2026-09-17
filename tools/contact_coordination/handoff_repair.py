"""Bounded source-conditioned handoff contact-pair construction on TRAIN.

Candidate object poses come from source carry prior and natural-configuration
FK, never a copied successful calibration world trajectory.
"""
from pathlib import Path
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json,atomic_npz
from .source_phase import COMMON,pose,mean_pose
from .morphology_repair import wrist_target,world_wrist,assign_named
from .planner import realize_phase_goals
from .runtime_hulls import Checker
from .prototype import INITIAL


def coupling_residual(left,right,enable_coupling):
    """The only factors removed in the contact-region coupling ablation."""
    factor=1. if enable_coupling else 0.
    return factor*np.r_[100*(left[:3,3]-right[:3,3]),
        10*Rotation.from_matrix(left[:3,:3].T@right[:3,:3]).as_rotvec()]


def carry_contact_for_candidate(carry,acquisition):
    """Keep the calibrated capture-to-carry relation under contact adaptation."""
    result=copy.deepcopy(carry)
    if 'acquisition_reference_T_HO' in carry:
        change=np.asarray(acquisition['T_HO'])@np.linalg.inv(carry['acquisition_reference_T_HO'])
        for field in ('T_HO','measured_T_HO'):
            if field in carry:result[field]=change@np.asarray(carry[field])
        result['candidate_adaptation']='G_acquisition_candidate * inverse(G_acquisition_calibration) * G_carry_calibration'
    return result


def run(out):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from .planning_kinematics import G1Kinematics
    c=load_common_config(COMMON);g1=G1Kinematics(c,load_scene(c));cal=read(out/'target_repair/CONTACT_CALIBRATION.json')
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id'];phase=read(out/'source_phase'/sid/'PHASE_RECORD.json');prior=dict(np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz'))
    x0=np.asarray(phase['initial_object_pose_world']);frames=phase['handoff_sample_indices'];raw=mean_pose(prior['inferred_object_from_left'][frames])
    contacts={'left':cal['contacts']['left_carry'],'right':cal['contacts']['handoff_right']}
    natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad']);g1.assign(natural)
    natural_objects={s:world_wrist(g1,s)@np.asarray(contacts[s]['T_wrist_H'])@np.asarray(contacts[s]['T_HO']) for s in contacts}
    center=np.mean([t[:3,3] for t in natural_objects.values()],axis=0)
    # The source object is only rigid-carry inferred. Keep its planar long-axis
    # direction, use gravity-upright support; no required roll/pitch was observed.
    long_axis=raw[:3,0];yaw=float(np.arctan2(long_axis[1],long_axis[0]))
    offsets={s:wrist_target(np.eye(4),contacts[s])[:3,3] for s in contacts}
    bilateral=offsets['right']-offsets['left']
    g1.assign(natural);robot_bilateral=world_wrist(g1,'right')[:3,3]-world_wrist(g1,'left')[:3,3]
    alignment_yaw=float(np.arctan2(robot_bilateral[1],robot_bilateral[0])-np.arctan2(bilateral[1],bilateral[0]))
    yaw_delta=float(np.arctan2(np.sin(alignment_yaw-yaw),np.cos(alignment_yaw-yaw)))
    poses=[pose(Rotation.from_euler('z',yaw+turn*yaw_delta).as_matrix(),(1-f)*raw[:3,3]+f*center)
           for f in (0.,.5,1.) for turn in (0.,.5,1.)]
    checker=Checker(g1,out,cal['joint_names']);config=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=120)
    lifted=np.load(out/'prototype'/sid/'morphology_acquisition_v4/PHASE_Q.npz')['q'][-1]
    hand=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q']);hand[:7]=contacts['left']['measured_finger_q'][:7]
    separation=np.asarray(read(out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json')['offset_object_frame_m'])
    source_deviation=[];bank={'left':[],'right':[]}
    for candidate,x in enumerate(poses):
        for side in bank:
            for fraction in ((0.,) if side=='left' else (0.,.5,1.)):
                target=wrist_target(x,contacts[side]);target[:3,3]+=x[:3,:3]@separation*fraction
                goal=dict(name='HANDOFF_'+side,active_hands=[side],wrist_pose_world={side:target},position_tolerance_m=.003,orientation_tolerance_rad=.05)
                r=realize_phase_goals(g1,[goal],lifted,config,seed_postures=[cal['contacts']['handoff_'+side]['seed_q']]);q=r.pop('q')[-1]
                unary=float(np.linalg.norm(x[:3,3]-raw[:3,3])**2+.01*Rotation.from_matrix(x[:3,:3].T@Rotation.from_euler('z',yaw).as_matrix()).magnitude()**2+np.linalg.norm(separation*fraction)**2)
                adapted_contact=copy.deepcopy(contacts[side]);adapted_contact['T_HO']=np.linalg.inv(target@np.asarray(adapted_contact['T_wrist_H']))@x
                bank[side].append(dict(candidate=candidate,contact_fraction=fraction,contact=adapted_contact,object_pose=x,q=q,unary_cost=unary,solve=r,wrist_target=target))
    selections={}
    for enabled in (True,False):
        pairs=[]
        # Same nine pair evaluations and IK bank for on/off; only cross-hand
        # consistency eligibility changes during selection, then common safety.
        for li,l in enumerate(bank['left']):
            for ri,r in enumerate(bank['right']):
                q=lifted.copy();q[:7]=l['q'][:7];q[7:14]=r['q'][7:14];f=hand.copy();f[7:]=contacts['right']['measured_finger_q'][7:]
                hits=getattr(checker,'query',checker.check)(np.r_[q,f],l['object_pose'],('left','right'))
                disagree=float(np.linalg.norm(np.asarray(l['object_pose'])[:3,3]-np.asarray(r['object_pose'])[:3,3]))
                valid=all(v['solve']['phases'][0]['goal_satisfied'] for v in [l,r])
                pairs.append(dict(left=li,right=ri,q=q,unary=l['unary_cost']+r['unary_cost'],disagreement_m=disagree,
                    cross_consistent=disagree<=.003,ik_valid=valid,forbidden=[h for h in hits if not h['allowed_contact']],
                    selectable=valid and not any(not h['allowed_contact'] for h in hits) and (not enabled or disagree<=.003)))
        admissible=[p for p in pairs if p['selectable']]
        chosen=min(admissible,key=lambda p:(p['unary'],p['left'],p['right'])) if admissible else None
        # Collision validity is common; no recoupling rescue after independent selection.
        selections[str(enabled)]=dict(selected=chosen,pairs=pairs,enable_coupling=enabled)
    folder=out/'prototype'/sid/'morphology_handoff_v5'/record(__file__)['sha256'][:12]
    atomic_json(folder/'CANDIDATES.json',dict(raw_source_object_prior=raw,natural_fk_object_center=center,poses=poses,bank=bank,selections=selections,
        bounds_provenance='Nine candidates: three positions from source-inferred carry center toward natural-q0 implied contact center, three yaw angles from source toward alignment of calibrated contact separation with the natural bilateral wrist axis. Actual full-SE3 IK and geometry determine admission.',
        orientation_permission='Source dynamic object orientation unobserved and no absolute handoff yaw requirement. Gravity-upright object; source planar long-axis yaw is a unary prior. Calibrated object-relative contact patches are preserved under common object rotation.',
        alignment_yaw=alignment_yaw,source_yaw=yaw,contact_separation_candidate=record(out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json'),implementation=record(__file__),
        source_phase=record(out/'source_phase'/sid/'PHASE_RECORD.json'),calibration=record(out/'target_repair/CONTACT_CALIBRATION.json'),config=config,
        enable_coupling_semantics='Cross-hand object-pose consistency during contact-pair selection only; same unary bank, kinematics, geometry and total evaluations.'))
    for flag,s in selections.items():
        p=s['selected'];print('HANDOFF',flag,'pair',None if p is None else [p['left'],p['right']], 'collision_count',None if p is None else len(p['forbidden']),flush=True)
    atomic_json(folder/'RESULT.json',dict(status='HANDOFF_CANDIDATES_CONSTRUCTED',selected_pairs={k:None if v['selected'] is None else [v['selected']['left'],v['selected']['right']] for k,v in selections.items()},physical_run=False))
    return selections


def _legacy_fit_region(out,contact_mode='translation',calibration_path=None,orientation_mode='upright',receiver_contact='handoff_right'):
    """Reuse the existing least-squares contact consistency formulation.

This replaces discrete absolute-position projection with a constrained region
fit. All non-coupling factors and the bounded candidate/seed bank are shared.
"""
    import time
    from scipy.optimize import least_squares
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from .planning_kinematics import G1Kinematics
    c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    calibration_path=Path(calibration_path) if calibration_path is not None else out/'target_repair/CONTACT_CALIBRATION.json'
    cal=read(calibration_path);phase=read(out/'source_phase'/sid/'PHASE_RECORD.json');prior=dict(np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz'))
    raw=mean_pose(prior['inferred_object_from_left'][phase['handoff_sample_indices']]);natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad'])
    # The giver already holds the incoming object. A relation measured after
    # receiver-induced deflection cannot stand in for that incoming contact.
    contacts={'left':cal['contacts']['left_carry'],'right':cal['contacts'][receiver_contact]}
    if receiver_contact=='receiver_acquisition_intent' and 'giver_handoff_intent' in cal['contacts']:
        contacts['left']=cal['contacts']['giver_handoff_intent']
    offset=np.zeros(3)
    preload_path=out/'target_repair/RECEIVER_CONTACT_PRELOAD_REGION.json'
    if contact_mode=='thumb_preload':
        preload=read(preload_path);offset=2*preload['offset_m']*np.asarray(preload['direction_object_frame'])
    elif contact_mode!='fixed_calibrated':
        offset=np.asarray(read(out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json')['offset_object_frame_m'])
    from .calibration_parameters import parameters
    tuning=parameters(out)
    offset=offset*tuning['receiver_spread_fraction']
    giver_seed=contacts['left'] if 'giver_handoff_intent' in cal['contacts'] else cal['contacts']['handoff_left']
    seeds=[natural,np.r_[giver_seed['seed_q'][:7],contacts['right']['seed_q'][7:]]]
    hand=np.r_[contacts['left']['measured_finger_q'][:7],contacts['right']['measured_finger_q'][7:]]
    checker=Checker(g,out,cal['joint_names']);bounds=g.arm_limits
    from .io import fingerprint
    gravity_path=out/'target_repair'/('CALIBRATED_RIGHT_CARRY_GRAVITY.json' if receiver_contact=='right_carry' else 'CALIBRATED_HANDOFF_GRAVITY.json')
    capture=cal.get('receiver_capture_transition') if receiver_contact=='receiver_acquisition_intent' else None
    if capture:
        gravity_path=calibration_path
        gravity={'up_direction_object_frame':capture['gravity_up_in_preobject']} if orientation_mode=='calibrated_gravity' else None
    else:gravity=read(gravity_path) if orientation_mode=='calibrated_gravity' else None
    capture_transform=np.asarray(capture['T_preobject_postobject']) if capture else np.eye(4)
    object_up=np.asarray(gravity['up_direction_object_frame']) if gravity is not None else np.array([0.,0.,1.])
    receiving_region=None
    if receiver_contact=='receiver_acquisition_intent':
        from .receiving_relation import build_region,realize_contact
        from .runtime_hulls import object_dimensions
        dimensions=object_dimensions(out)
        receiving_region=build_region(phase,contacts['right'],dimensions,
            read(out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json')['offset_object_frame_m'],object_up)
        contacts['right']=realize_contact(contacts['right'],receiving_region)
    from .scientific_cache import key as cache_key
    key=cache_key(out,sid,'receiving_candidate_and_coupled_selection',
        dict(contact_mode=contact_mode,orientation_mode=orientation_mode,receiver_contact=receiver_contact),
        [calibration_path]+([preload_path] if contact_mode=='thumb_preload' else []))
    folder=out/'prototype'/sid/'contact_region_morphology'/(key[:12]+'_'+contact_mode+'_'+orientation_mode+'_'+receiver_contact)
    if (folder/'RESULT.json').exists():
        receipt=read(folder/'CACHE_CONTRACT.json')
        if receipt['key']!=key:raise ValueError('Contact cache identity mismatch')
        for artifact in receipt['artifacts']:
            if record(artifact['path'])!=artifact:raise ValueError('Changed cached contact result')
        atomic_json(out/'target_repair/CURRENT_CONTACT_REGION_FIT.json',dict(path=str(folder.resolve()),cache_key=key,result=record(folder/'RESULT.json'),config=record(folder/'CONFIG.json')))
        return {str(flag):read(folder/f'coupling_{flag}.json') for flag in (True,False)}
    if folder.exists():raise FileExistsError('Incomplete immutable contact fit; preserve and diagnose: '+str(folder))
    contact_angle=2*np.arcsin(np.linalg.norm(offset)/.0725)
    fractions=(0.,) if contact_mode=='fixed_calibrated' else (0.,.5,1.)
    if contact_mode=='translation':
        from .contact_candidate_region import translation_bank
        contact_bank=translation_bank(offset)
    else:
        contact_bank=[dict(offset=offset*f,priority=0,family=contact_mode,fraction=f) for f in fractions]
    config=dict(max_nfev=180,seeds=2,contact_candidates=len(contact_bank),source_position_scale=1.,source_yaw_scale=.1,
        upright_axis_scale=10.,cross_position_scale=100.,cross_rotation_scale=10.,region_scale=100.,
        position_consistency_m=.003,rotation_consistency_rad=.05,joint_posture_scale=.1,
        posture_prior_basis='Bounded TRAIN calibration sensitivity diagnosis:100mm-equivalent source-position cost per radian. The10mm/rad version still displaced redundant shoulder/wrist rotations from the qualified loaded grasp and lost retention after giver clearance. Keep source pose priors, region, contacts and budget unchanged; test the stronger common seed prior. No DEV outcome used.')
    config['joint_posture_scale']*=tuning['handoff_posture_multiplier']
    config['bounded_calibration_parameters']=tuning
    atomic_json(folder/'CONFIG.json',dict(config=config,source_phase=record(out/'source_phase'/sid/'PHASE_RECORD.json'),calibration=record(calibration_path),implementation=record(__file__),
        source_receiving_region=receiving_region,scientific_cache_key=key,
        position_region='Object entirely above table at qualified 63 mm lift clearance, within table XY. Upper Z is source handoff prior height; actual named IK/collider validity is checked, not inferred from a shoulder sphere.',
        orientation_region='Calibrated gravity direction in the contact/object frame is preserved; source yaw remains a soft prior. Absolute source handoff roll/pitch is UNKNOWN.' if gravity else 'Object up axis constrained to gravity; yaw follows the source prior as a soft cost because dynamic source orientation and an absolute task-required handoff yaw are UNKNOWN.',
        orientation_mode=orientation_mode,receiver_contact=receiver_contact,object_up_direction=object_up,gravity_calibration=record(gravity_path) if gravity else None,
        contact_preload_region=record(preload_path) if contact_mode=='thumb_preload' else None,
        receiver_capture_transition=capture,
        seeds=seeds,contact_mode=contact_mode,contact_offsets=[v['offset'] for v in contact_bank],
        contact_bank=contact_bank,
        contact_region_provenance='Existing calibrated displacement radius; original clearance-line samples followed by object-box face and face-diagonal directions. Full contact/closing geometry remains mandatory. No world pose or outcome is a candidate input.',
        contact_yaw_angles_rad=[0.,float(contact_angle),float(-contact_angle)] if contact_mode=='rotation' else [0.]*len(contact_bank),
        rotation_bound_provenance='Chord clearance from measured calibration thumb-hull overlap + existing contact offset, using proxy short-axis radius. Rotation about object center retains radial enclosure; contact candidates are not yet physically qualified.',
        enable_coupling_only='cross-hand predicted object position/rotation residuals',frozen=False))
    if receiving_region is not None:atomic_json(folder/'RECEIVING_REGION.json',receiving_region)
    results={};yaw=np.arctan2(raw[1,0],raw[0,0]);sourceR=Rotation.from_euler('z',yaw).as_matrix()@Rotation.align_vectors([[0.,0.,1.]],[object_up])[0].as_matrix()
    for enabled in (True,False):
        rows=[]
        for ci,contact_proposal in enumerate(contact_bank):
            displacement=np.asarray(contact_proposal['offset'])
            cc=copy.deepcopy(contacts);t=wrist_target(np.eye(4),cc['right'])
            if contact_mode in ('translation','thumb_preload'):t[:3,3]+=displacement
            elif contact_mode!='fixed_calibrated':
                angle=(0.,contact_angle,-contact_angle)[ci];turn=pose(Rotation.from_euler('z',angle).as_matrix(),np.zeros(3));t=turn@t
            cc['right']['T_HO']=np.linalg.inv(t@np.asarray(cc['right']['T_wrist_H']))
            relations={s:np.asarray(cc[s]['T_wrist_H'])@np.asarray(cc[s]['T_HO']) for s in cc}
            def objects(q):
                g.assign(q);return {s:world_wrist(g,s)@relations[s] for s in cc}
            for si,seed in enumerate(seeds):
                started=time.monotonic()
                def residual(q):
                    xs=objects(q);value=[]
                    for x in xs.values():
                        half=np.abs(x[:3,:3])@np.asarray([.1175,.0725,.0775])/2
                        value.extend(x[:3,3]-raw[:3,3]);value.extend(.1*Rotation.from_matrix(sourceR.T@x[:3,:3]).as_rotvec())
                        value.extend(10*(x[:3,:3]@object_up-[0,0,1]))
                        lo=np.array([0,0,.795+.063])+half;hi=np.array([.835,.72,max(raw[2,3],.795+.063+.0775)])-half
                        value.extend(100*np.maximum(lo-x[:3,3],0));value.extend(100*np.maximum(x[:3,3]-hi,0))
                    value.extend(coupling_residual(xs['left'],xs['right'],enabled))
                    value.extend(config['joint_posture_scale']*(q-seed));return np.asarray(value)
                sol=least_squares(residual,np.minimum(np.maximum(seed,bounds[:,0]+1e-7),bounds[:,1]-1e-7),bounds=(bounds[:,0]+1e-7,bounds[:,1]-1e-7),max_nfev=config['max_nfev'],ftol=1e-9,xtol=1e-9,gtol=1e-9)
                xs=objects(sol.x);dp=float(np.linalg.norm(xs['left'][:3,3]-xs['right'][:3,3]));dr=float(Rotation.from_matrix(xs['left'][:3,:3].T@xs['right'][:3,:3]).magnitude())
                overlap_contacts=copy.deepcopy(cc)
                for side in cc:
                    overlap_contacts[side]['T_HO']=np.asarray(cc[side]['T_HO'])@capture_transform
                    if capture:
                        overlap_contacts[side]['measured_finger_q']=(cc[side]['measured_finger_q'] if side=='left' else cal['contacts']['handoff_'+side]['measured_finger_q'])
                        overlap_contacts[side]['evidence']='CALIBRATED_COMMAND_SIDE_CAPTURE_PREDICTION_NOT_MEASURED_RIGID_CONTACT'
                overlap_objects={side:xx@capture_transform for side,xx in xs.items()}
                overlap_hand=np.r_[overlap_contacts['left']['measured_finger_q'][:7],overlap_contacts['right']['measured_finger_q'][7:]]
                hits=getattr(checker,'query',checker.check)(np.r_[sol.x,hand],xs['left'],('left','right'))
                hits+=getattr(checker,'query',checker.check)(np.r_[sol.x,overlap_hand],overlap_objects['left'],('left','right'))
                bad=[h for h in hits if not h['allowed_contact']]
                from .contact_transition_geometry import check_receiver_closing
                closing=check_receiver_closing(checker,sol.x,overlap_contacts['left']['measured_finger_q'],overlap_objects['left'])
                bad.extend(closing['forbidden_contacts'])
                upright=max(float(np.linalg.norm(x[:3,:3]@object_up-[0,0,1])) for x in xs.values())
                valid=not bad and dp<=.003 and dr<=.05 and upright<=.05
                row=dict(contact_candidate=ci,seed=si,q=sol.x,objects=xs,contacts=cc,measured_hand_prediction=hand,
                    candidate_priority=contact_proposal['priority'],contact_region_proposal=contact_proposal,
                    carry_contacts={'left':cal['contacts']['left_carry'],'right':carry_contact_for_candidate(cal['contacts'].get('right_carry_command_intent',cal['contacts']['right_carry']),cc['right'])},
                    overlap_objects=overlap_objects,overlap_contacts=overlap_contacts,receiver_capture_transition=capture,
                    position_disagreement_m=dp,rotation_disagreement_rad=dr,upright_error=upright,forbidden=bad,
                    receiver_closing_geometry=closing,
                    valid_stationary_overlap=valid,full_path_valid=False,runtime_s=time.monotonic()-started,nfev=int(sol.nfev),cost=float(sol.cost)+float(displacement@displacement),contact_preload_m=float(np.linalg.norm(displacement)))
                rows.append(row);print('REGION',enabled,ci,si,'valid',valid,'dp',round(dp,5),'collisions',len(bad),flush=True)
        results[str(enabled)]=rows;atomic_json(folder/f'coupling_{enabled}.json',rows)
    atomic_json(folder/'RESULT.json',dict(status='CONTACT_REGION_FIT_COMPLETE',valid_counts={k:sum(r['valid_stationary_overlap'] for r in v) for k,v in results.items()},physical_run=False))
    atomic_json(folder/'CACHE_CONTRACT.json',dict(key=key,stage='receiving_candidate_and_coupled_selection',artifacts=[record(folder/name) for name in ('CONFIG.json','coupling_True.json','coupling_False.json','RESULT.json')]))
    atomic_json(out/'target_repair/CURRENT_CONTACT_REGION_FIT.json',dict(path=str(folder.resolve()),cache_key=key,result=record(folder/'RESULT.json'),config=record(folder/'CONFIG.json')))
    return results


def fit_region(out,contact_mode='translation',calibration_path=None,orientation_mode='calibrated_gravity',receiver_contact='receiver_acquisition_intent'):
    from .shared_handoff_candidates import build
    return build(out,contact_mode,calibration_path,orientation_mode,receiver_contact)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--region-fit',action='store_true');p.add_argument('--contact-mode',choices=['translation','rotation','fixed_calibrated','thumb_preload'],default='translation');p.add_argument('--calibration',type=Path);p.add_argument('--orientation-mode',choices=['upright','calibrated_gravity'],default='upright');p.add_argument('--receiver-contact',choices=['handoff_right','right_carry','receiver_acquisition_intent'],default='handoff_right');a=p.parse_args();fit_region(a.run_dir,a.contact_mode,a.calibration,a.orientation_mode,a.receiver_contact) if a.region_fit else run(a.run_dir)
