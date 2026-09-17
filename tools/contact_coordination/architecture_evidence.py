"""Read executable artifacts into a complete candidate/connection evidence ledger."""
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json


def handoff_translation_test(out,golden):
    """Synthetic registered-task translation, never a new source or result."""
    import shutil
    from .scientific_cache import digest,CODE_AT_PROCESS_IMPORT
    from .shared_handoff_candidates import build
    sid=golden['source_id'];context=Path(golden['context']);baseline=Path(read(context/'target_repair/CURRENT_CONTACT_REGION_FIT.json')['path'])
    delta=np.array([.008,-.004,0.]);token=digest(dict(code=CODE_AT_PROCESS_IMPORT,test=record(__file__),baseline=record(baseline/'ENDPOINT_BANK.json'),delta=delta))[:12]
    fixture=out/'structural_fixtures'/('handoff_translation_'+token);receipt=fixture/'TRANSLATION_RESULT.json'
    if receipt.exists():
        prior=read(receipt)
        if all(record(r['path'])==r for r in prior['artifacts']):return record(receipt)
        raise AssertionError('Changed completed translation fixture')
    for relative in ('target_repair','source_phase/'+sid):
        source=context/relative
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.npz') and path.name!='CURRENT_CONTACT_REGION_FIT.json':
                target=fixture/relative/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for relative in ('bootstrap/SELECTION.json','CALIBRATION_PARAMETERS.json','SPLIT_CONTRACT.json'):
        target=fixture/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(context/relative,target)
    phase_path=fixture/'source_phase'/sid/'PHASE_RECORD.json';phase=read(phase_path)
    phase['initial_object_pose_world']=np.asarray(phase['initial_object_pose_world']);phase['initial_object_pose_world'][:3,3]+=delta
    phase['synthetic_registered_task_translation_for_structural_test_only']=delta
    atomic_json(phase_path,phase)
    from .io import atomic_npz
    for name in ('SOURCE_PRIORS.npz','FUNCTIONAL_WRIST_PRIORS.npz'):
        path=fixture/'source_phase'/sid/name;values=dict(np.load(path))
        for field,array in values.items():
            if array.ndim>=2 and array.shape[-2:]==(4,4):array[...,:3,3]+=delta
            elif field.endswith('_position_world') and array.ndim==2 and array.shape[1]==3:array+=delta
        atomic_npz(path,**values)
    build(fixture)
    moved=Path(read(fixture/'target_repair/CURRENT_CONTACT_REGION_FIT.json')['path'])
    a=read(baseline/'ENDPOINT_BANK.json');b=read(moved/'ENDPOINT_BANK.json')
    raw_delta=np.asarray(b['raw_prior'])[:3,3]-np.asarray(a['raw_prior'])[:3,3]
    center_delta=np.asarray(b['region_center'])-np.asarray(a['region_center'])
    np.testing.assert_allclose(raw_delta,delta,atol=1e-12,rtol=0)
    assert center_delta@delta>0 and 1e-6<np.linalg.norm(center_delta)<4*np.linalg.norm(delta)
    assert not np.allclose(a['region_poses'],b['region_poses'])
    atomic_json(receipt,dict(status='PASS',source_id=sid,synthetic_fixture_not_dataset=True,
        translated_task_condition_m=delta,raw_prior_shift_m=raw_delta,reachable_region_center_shift_m=center_delta,
        interpretation='Reachability projection responds in the source translation direction; robot/table constraints remain fixed, so exact rigid translation of all feasible endpoints is not required.',
        candidate_budgets_unchanged=a['fixed_budget']==b['fixed_budget'],
        artifacts=[record(baseline/'ENDPOINT_BANK.json'),record(moved/'ENDPOINT_BANK.json'),record(phase_path),
            record(fixture/'source_phase'/sid/'SOURCE_PRIORS.npz'),record(__file__)]))
    return record(receipt)


def distinct(values):
    def key(value):
        if isinstance(value,dict):return tuple((s,key(v)) for s,v in sorted(value.items()))
        return np.asarray(value).round(10).tobytes()
    return len({key(v) for v in values})


def collect(out):
    from .wrist_reference import trajectory_reference
    rows=read(out/'GOLDEN_PLANNING.json')['rows']+read(out/'COVERAGE8_PLANNING.json')['rows']
    ledger=[];chains=[];banks={};reference=[]
    bounds=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][:14]
    velocity=np.array([v['max_velocity_rad_s'] for v in bounds]);acceleration=np.array([v['max_acceleration_rad_s2'] for v in bounds])
    for result in rows:
        sid=result['source_id'];method=result['method_key'];context=Path(result['context'])
        grasp=read(Path(result['receipt']).parent/'GRASP_CANDIDATES.json')
        for phase in ('PREGRASP','LEFT_ACQUISITION','LIFT'):
            for candidate in grasp:
                branch=next((b for b in result['attempts'] if b['candidate_id']==candidate['candidate_id']),None)
                endpoint=None
                if branch:
                    plan=Path(branch['context'])/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json'
                    endpoint=next((p for p in read(plan)['phases'] if p['phase']==phase),None)
                ik=any(c['goal_satisfied'] for c in endpoint['candidates']) if endpoint else None
                geometry=any(c['admissible'] for c in endpoint['candidates']) if endpoint else None
                ledger.append(dict(candidate_id=candidate['candidate_id']+':'+phase,source_id=sid,method=method,phase=phase,
                    source_relation=candidate['source_relation'],local_perturbation=candidate['local_perturbation'],
                    mode_seed=candidate['mode_seed'],task_space_target=candidate['targets'][phase],
                    IK_result=ik,geometry_result=geometry,planner_result=endpoint['admissible'] if endpoint else None,
                    score=candidate['score'],selected=result.get('selected_candidate_id')==candidate['candidate_id'],
                    not_evaluated_reason=None if endpoint else 'Earlier branch/phase pruned or a higher-ranked complete chain selected'))
        pointer=context/'target_repair/CURRENT_CONTACT_REGION_FIT.json'
        if pointer.exists():
            folder=Path(read(pointer)['path']);bank=read(folder/'ENDPOINT_BANK.json')['bank']
            banks[(sid,method)]=(bank,folder,result)
            attempted_pairs=[]
            for branch in result['attempts']:
                if branch.get('connection'):
                    summary=read(Path(branch['connection'])/'RESULT.json')
                    attempted_pairs.extend(summary['results'])
            ranked=read(folder/f'coupling_{method=="C_COUPLED"}.json')
            selected=read(Path(result['plan'])/'CONTACT_SELECTION.json')['selected_candidate_ids'] if result['full_task_plan'] else []
            for side,values in bank.items():
                for candidate in values:
                    related=[a for a in attempted_pairs if any(v['contact_candidate']==a['contact_candidate'] and candidate['candidate_id'] in v['selected_candidate_ids'] for v in ranked)]
                    ledger.append(dict(candidate,method=method,selected=candidate['candidate_id'] in selected,
                        planner_result=any(r['completed_phases']>=2 for r in related) if related else None,
                        complete_chain_connectable=any(r['all_admissible'] for r in related) if related else None))
        if not result['full_task_plan']:continue
        connection=Path(result['plan']).parent;summary=read(connection/'RESULT.json')
        selected=next(r for r in summary['results'] if r['all_admissible']);sub=connection/selected['subdirectory']
        late=read(sub/'PHASE_IK.json');early=read(context/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json')
        goals=read(sub/'GOALS.json');phases=early['phases']+late['phases']
        certificates=[]
        for phase in phases:
            preparation=phase.get('passive_hand_preparation')
            parts=next(a['result']['phases'] for a in preparation['attempts'] if a['valid']) if preparation else [phase]
            joined=[];proofs=[]
            for part in parts:
                accepted=[p for p in part.get('planner_attempts',[]) if p['status']=='PATH_FOUND']
                if not accepted:raise AssertionError('Selected subconnection has no path certificate: '+part['phase'])
                certificate=accepted[-1]
                assert part['connection_method']=='COLLISION_AWARE_GLOBAL_PLANNER'
                assert certificate['state_checks']>0 and certificate['edge_checks']>0
                assert certificate['budget']['edge_resolution_rad']==.005
                np.testing.assert_allclose(part['connecting_q'],certificate['path'])
                if joined:np.testing.assert_allclose(joined[-1],certificate['path'][0])
                joined.extend(certificate['path'][1:] if joined else certificate['path'])
                proofs.append(dict(phase=part['phase'],search_used=certificate['search_used'],direct_path_valid=certificate['direct_path_valid'],
                    state_checks=certificate['state_checks'],edge_checks=certificate['edge_checks'],budget=certificate['budget']))
            np.testing.assert_allclose(phase['connecting_q'],joined)
            certificates.append(dict(phase=phase['phase'],search_used=any(p['search_used'] for p in proofs),
                direct_path_valid=proofs[0]['direct_path_valid'] if len(proofs)==1 else None,
                state_checks=sum(p['state_checks'] for p in proofs),edge_checks=sum(p['edge_checks'] for p in proofs),
                budget=proofs[0]['budget'],subconnections=proofs))
        def region(search):
            if not isinstance(search,dict):return
            values=search.get('generated',[])
            from scipy.spatial.transform import Rotation
            from .scientific_cache import digest
            prior=values[0]['task_space_target'] if values else {}
            for index,item in enumerate(values):
                local={side:dict(translation_in_prior_wrist_frame_m=np.asarray(prior[side])[:3,:3].T@(
                    np.asarray(target)[:3,3]-np.asarray(prior[side])[:3,3]),
                    relative_rotation_vector_rad=Rotation.from_matrix(np.asarray(prior[side])[:3,:3].T@np.asarray(target)[:3,:3]).as_rotvec())
                    for side,target in item['task_space_target'].items()}
                yield dict(item,source_id=sid,method=method,
                    active_candidate_id=item['candidate_id'],candidate_id=sid+':'+item['phase']+':'+digest(item['task_space_target'])[:12],
                    selected=item['candidate_id']==search.get('selected_candidate_id'),
                    source_relation=item.get('source_relation') or read(context/'source_phase'/sid/'PHASE_RECORD.json')['source_functional_tool_object_relations'],
                    local_perturbation=item.get('local_perturbation') or local,mode_seed='BOUNDED_GOAL_REGION_ENDPOINT_SEEDS',
                    score=item.get('score') if item.get('score') is not None else dict(geometric_region_priority=index,
                        ordering='Active deterministic region order, followed by IK/geometry/connection feasibility. No physics outcome used.'))
            # Retained hierarchical refinements are genuine generated proposals.
            for name,value in search.items():
                if name not in ('attempts','generated','goal') and isinstance(value,dict):yield from region(value)
        for field in ('receiver_departure_search','transport_region_search','placement_region_search'):
            ledger.extend(region(late.get(field)))
        for phase in late['phases']:ledger.extend(region(phase.get('target_region_selection')))
        command=np.load(Path(result['plan'])/'COMMANDS.npz');q=command['commanded_q_rad'][:,:14]
        v=np.diff(q,axis=0)*30.;a=np.diff(v,axis=0)*30.
        assert np.all(np.max(abs(v),axis=0)<=velocity+1e-6)
        assert np.all(np.max(abs(a),axis=0)<=acceleration+1e-6)
        labels=command['stage'].astype(str)
        assert np.flatnonzero(labels=='RIGHT_OWNERSHIP_VERIFY')[0]<np.flatnonzero(labels=='RECEIVER_DEPARTURE')[0]
        closing=np.isin(labels,['PRESHAPE','POWER_GRASP'])
        assert np.max(np.ptp(q[closing],axis=0))<1e-12
        assert np.all(command['common_task_intent'][labels=='ACQUISITION_INGRESS']=='OPEN_INTENT')
        checked=read(Path(result['plan'])/'FINAL_COMMAND_VALIDATION.json');assert checked['status']=='VALID'
        chains.append(dict(source_id=sid,method=method,certificates=certificates,
            stationary_acquisition_closing=True,ownership_precedes_departure=True,
            retimed_velocity_acceleration_pass=True,post_retime_collision_pass=True,plan=record(Path(result['plan'])/'PLAN.json')))
    # Actual identical-bank evidence, not merely a synthetic selector example.
    bc=[]
    for sid in sorted({s for s,m in banks}):
        if not all((sid,m) in banks for m in ('B_INDEPENDENT','C_COUPLED')):continue
        b,bpath,_=banks[(sid,'B_INDEPENDENT')];c,cpath,_=banks[(sid,'C_COUPLED')]
        same=all(len(b[s])==len(c[s]) and all(x['candidate_id']==y['candidate_id'] and
            np.array_equal(x['task_space_target'],y['task_space_target']) and np.array_equal(x['q'],y['q']) and
            x['unary_score']==y['unary_score'] for x,y in zip(b[s],c[s])) for s in ('left','right'))
        rb=read(bpath/'RANKING_False.json');rc=read(cpath/'RANKING_True.json')
        assert same and all(r['coupling_weight']==0 and r['shared_object_pose'] is None and r['score']==r['unary_score'] for r in rb)
        assert all(r['coupling_weight']==1 and r['shared_object_pose'] is not None for r in rc)
        changed=sum(r['cross_hand_score']>1e-8 for r in rc);assert changed>0
        bc.append(dict(source_id=sid,identical_endpoint_bank=True,pairs_with_changed_coupled_cost=changed,
            independent_top_ids=[rb[0]['left_candidate_id'],rb[0]['right_candidate_id']],
            coupled_top_ids=[rc[0]['left_candidate_id'],rc[0]['right_candidate_id']],evidence=[record(bpath/'ENDPOINT_BANK.json'),record(cpath/'RANKING_True.json')]))
    assert bc
    # A remains a source-curve representation and does not share contact ranking.
    for sid in read(out/'COVERAGE8.json')['source_ids'][:3]:
        source=out/'source_phase'/sid
        samples=trajectory_reference(source,'left','LEFT_TRANSPORT_BEGIN','RIGHT_APPROACH_BEGIN')
        assert len(samples)>2 and all(s['source_input']=='REGISTERED_SOURCE_WRIST_TRAJECTORY' for s in samples)
        reference.append(dict(source_id=sid,source_wrist_keyframes=len(samples),representation='WRIST_REFERENCE'))
    multiplicity=[]
    for chain in chains:
        for phase in ('PREGRASP','LEFT_ACQUISITION','LIFT','HANDOFF_LEFT','HANDOFF_RIGHT','RECEIVER_DEPARTURE','PLACE'):
            values=[r for r in ledger if r['source_id']==chain['source_id'] and r['method']==chain['method'] and r['phase']==phase]
            count=distinct([r['task_space_target'] for r in values])
            if count<2:raise AssertionError('Effective phase multiplicity missing: '+str((chain['source_id'],chain['method'],phase,count)))
            multiplicity.append(dict(source_id=chain['source_id'],method=chain['method'],phase=phase,distinct_task_targets=count))
    translation=handoff_translation_test(out,read(out/'GOLDEN_PLANNING.json')['rows'][0])
    path=out/'ACTIVE_IMPLEMENTATION_EVIDENCE.json';candidate_path=out/'REPAIRED_CANDIDATE_LEDGER.json'
    atomic_json(candidate_path,dict(candidates=ledger,not_evaluated_is_not_a_pass=True))
    atomic_json(path,dict(status='PASS',chains=chains,BC_actual_bank_regression=bc,A_trajectory_reference_regression=reference,
        phase_multiplicity=multiplicity,handoff_translation_test=translation,ledger=record(candidate_path),no_physical_success_oracle=True))
    return [path,candidate_path,Path(translation['path'])]
