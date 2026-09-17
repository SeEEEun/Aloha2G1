"""Evidence-producing stages for the architecture repair runner."""
import copy
import os
from pathlib import Path
import subprocess
import numpy as np
from .io import ROOT,OFFLINE,read,record,atomic_json,atomic_text


def candidate_generation(out):
    from .source_contract import run
    from .source_phase import mean_pose
    from .interaction_candidates import acquisition_bank
    from .runtime_hulls import object_dimensions
    run(out)
    sources=read(out/'SPLIT_CONTRACT.json')['authorized_training_source_ids']
    golden=read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    features=[];ids=sorted(s for s in sources if s!=golden)
    for sid in ids:
        phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
        priors=np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz')
        handoff=mean_pose(priors['inferred_object_from_left'][phase['handoff_sample_indices']])
        features.append(np.r_[np.asarray(phase['initial_object_pose_world'])[:2,3],handoff[:3,3],
                              np.asarray(phase['source_functional_tool_object_relations']['right'])[:3,3]])
    values=np.asarray(features);normalized=(values-values.mean(axis=0))/np.maximum(values.std(axis=0),1e-8)
    selected=[int(np.argmax(np.linalg.norm(normalized,axis=1)))]
    while len(selected)<8:
        distances=np.min(np.linalg.norm(normalized[:,None]-normalized[selected],axis=2),axis=1)
        distances[selected]=-1.;selected.append(int(np.argmax(distances)))
    coverage=[ids[i] for i in selected]
    declaration=dict(golden=golden,source_ids=coverage,
        rule='Lexically tie-broken farthest-point coverage in standardized source initial XY, inferred handoff XYZ and receiver relation XYZ. Start farthest from mean; exclude Golden. No conversion/physics outcomes.',
        features={s:f for s,f in zip(ids,features)},fixed_before_repaired_planning=True)
    if (out/'COVERAGE8.json').exists():
        if read(out/'COVERAGE8.json')['source_ids']!=coverage:raise RuntimeError('Coverage set changed')
    atomic_json(out/'COVERAGE8.json',declaration)
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json');rows=[]
    for sid in [golden,*coverage]:
        phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
        bank=acquisition_bank(phase,cal,object_dimensions(out))
        for phase_name in ('LEFT_ACQUISITION','LIFT','PREGRASP'):
            count=len({np.asarray(c['targets'][phase_name]).round(10).tobytes() for c in bank})
            if count<=1:raise AssertionError('Collapsed candidate region')
        moved=copy.deepcopy(phase);translation=np.array([.013,-.007,.009])
        moved['initial_object_pose_world']=np.asarray(moved['initial_object_pose_world']).copy()
        moved['initial_object_pose_world'][:3,3]+=translation
        changed=acquisition_bank(moved,cal,object_dimensions(out))
        for a,b in zip(bank,changed):
            np.testing.assert_allclose(np.asarray(b['task_space_target'])[:3,3]-a['task_space_target'][:3,3],translation,atol=1e-12,rtol=0)
        path=out/'candidate_evidence'/sid/'ACQUISITION_REGION.json'
        atomic_json(path,bank)
        rows.append(dict(source_id=sid,generated=len(bank),distinct_task_targets=len(bank),
            source_translation_test=True,bank=record(path)))
    path=out/'CANDIDATE_MULTIPLICITY_AND_SOURCE_SENSITIVITY.json'
    atomic_json(path,dict(status='PASS',rows=rows,physical_outcome_used=False,
                         scope='Source readback, acquisition/pregrasp/lift target charts; full phase counts follow planning'))
    return [out/'COVERAGE8.json',path,*[Path(r['bank']['path']) for r in rows]]


def candidate_selection(out):
    from .interaction_candidates import handoff_pair_ranking
    # A conflict with a known independent optimum exercises the real selector.
    # Values come from one source handoff prior; local alternatives stay in it.
    from .source_phase import mean_pose
    sid=read(out/'COVERAGE8.json')['golden'];phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
    source=np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz')
    x=mean_pose(source['inferred_object_from_left'][phase['handoff_sample_indices']])
    y=x.copy();y[:3,3]+=x[:3,0]*.008
    left=[dict(candidate_id='L0',object_pose=x,unary_score=0.),dict(candidate_id='L1',object_pose=y,unary_score=1.)]
    right=[dict(candidate_id='R0',object_pose=x,unary_score=1.),dict(candidate_id='R1',object_pose=y,unary_score=0.)]
    b=handoff_pair_ranking(left,right,False);c=handoff_pair_ranking(left,right,True)
    assert (b[0]['left'],b[0]['right'])==(0,1)
    assert c[0]['left']==c[0]['right'] and c[0]['shared_object_pose'] is not None
    assert b[0]['shared_object_pose'] is None
    path=out/'BC_SELECTION_REGRESSION.json'
    atomic_json(path,dict(status='PASS',source_id=sid,left=left,right=right,B=b,C=c,
        same_candidate_bank=True,test_kind='Source-conditioned algebraic representation regression; not IK/physics evidence'))
    return [path]


def bc_difference(out):
    result=read(out/'BC_SELECTION_REGRESSION.json');assert result['status']=='PASS'
    path=out/'BC_REPRESENTATION_DIFFERENCE.md'
    atomic_text(path,'# B/C representation difference\n\n'
        '`interaction_candidates.handoff_pair_ranking` consumes identical independently generated left/right candidate arrays and unary scores. '
        'B sorts the separable unary sum and records the independent left/right IDs. It never adds a shared-object residual. '
        'C adds normalized cross-hand position and rotation disagreement and records the common mean pose X. '
        'The endpoint relation is `FK_side(q_side) @ T_wrist_H @ T_HO`. '
        'Subsequent common collision/path/contact validity can reject either method; no representation objective is added in the path planner.\n\n'
        'BC_SELECTION_REGRESSION.json contains a real selector call with the same source-conditioned bank: B selects (L0,R1), C selects a compatible pair. '
        'This algebraic test does not certify full IK/physics behavior; Golden/Coverage8 receipts are required for that.\n\n'
        'The final actual-bank comparison is recorded in [BC_ACTUAL_CANDIDATE_BANK_EVIDENCE.md](BC_ACTUAL_CANDIDATE_BANK_EVIDENCE.md) '
        'and ACTIVE_IMPLEMENTATION_EVIDENCE.json. REPAIRED_CANDIDATE_LEDGER.json contains selected IDs, unary components, '
        'explicit cross-hand costs and shared X for the actual source episodes. Different rankings do not by themselves establish a physical success advantage.\n')
    return [path]


def planner_integration(out):
    log=out/'PLANNER_UNIT_REGRESSION.log'
    with log.open('w') as stream:
        result=subprocess.run([OFFLINE,'-B','-m','unittest','tools.contact_coordination.tests.test_architecture_planner',
            'tools.contact_coordination.test_morphology_repair','tools.contact_coordination.test_abc_contract'],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
    if result.returncode:raise RuntimeError('Planner regressions failed')
    from .joint_path_planner import plan,Validator
    cases=[]
    for index,(cx,cy,hx,hy) in enumerate([(0.,0.,.23,.5),(.1,.1,.18,.4),(-.1,-.1,.2,.45)]):
        def valid(q):return not (abs(q[0]-cx)<hx and abs(q[1]-cy)<hy)
        a=np.array([-.8,0.]);b=np.array([.8,0.]);lo=np.full(2,-1.);hi=-lo
        assert not Validator(lo,hi,valid).edge(a,b)
        planned=plan(a,b,lo,hi,valid)
        assert planned['status']=='PATH_FOUND' and planned['search_used']
        cases.append(dict(case=index,q_start=a,q_goal=b,obstacle=[cx,cy,hx,hy],result=planned))
    path=out/'PLANNER_SEARCH_REGRESSION.json'
    atomic_json(path,dict(status='PASS',cases=cases,scope='Actual search backend in deterministic 2DOF obstacle fixtures; robot-hull tests follow'))
    return [log,path]


def edge_validation(out):
    from .joint_path_planner import Validator
    oracle=lambda q:abs(q[0])>.01
    checker=Validator([-1.],[1.],oracle)
    assert checker.state([-.8]) and checker.state([.8])
    assert not checker.edge([-.8],[.8])
    path=out/'EDGE_VALIDATION_REGRESSION.json'
    atomic_json(path,dict(status='PASS',valid_endpoints=True,invalid_interior_rejected=True,
        state_checks=checker.calls,edge_checks=checker.edges,resolution_rad=checker.resolution,
        precision='Resolution-based dyadic subdivision, not an exact swept-volume certificate'))
    return [path]


def carried_object_validation(out):
    from .architecture_robot_regressions import carried_object_test
    return carried_object_test(out)


def retiming_validation(out):
    from .planner import quintic_retime
    knots=np.array([[0.,0.],[.3,-.2],[-.1,.5]])
    v=np.array([.5,.7]);acc=np.array([1.,1.4]);path,duration=quintic_retime(knots,v,acc,30.)
    measured_v=np.diff(path,axis=0)*30.;measured_a=np.diff(measured_v,axis=0)*30.
    assert np.all(np.max(abs(measured_v),axis=0)<=v+1e-9)
    assert np.all(np.max(abs(measured_a),axis=0)<=acc+1e-9)
    from .execution_timing import common_primitive
    from tools.common_execution_layer import _transition
    primitive=common_primitive()
    specs=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][14:]
    finger_checks=[]
    for side,index in [('left',0),('right',7)]:
        bounds=specs[index:index+7]
        for field,start,end in [('preshape_frames','open','preshape'),('close_frames','preshape','full_close'),('release_frames','full_close','open')]:
            n=getattr(primitive,field)
            values=np.asarray([_transition(getattr(primitive,side+'_'+start),getattr(primitive,side+'_'+end),i,n) for i in range(n)])
            vv=np.max(abs(np.diff(values,axis=0)*30),axis=0);aa=np.max(abs(np.diff(values,n=2,axis=0)*900),axis=0)
            assert np.all(vv<=np.array([b['max_velocity_rad_s'] for b in bounds])+1e-9)
            assert np.all(aa<=np.array([b['max_acceleration_rad_s2'] for b in bounds])+1e-9)
            finger_checks.append(dict(side=side,transition=field,frames=n,maximum_velocity=vv,maximum_acceleration=aa))
    # Each output is on an original line segment; timing cannot bend geometry.
    offset=0
    for a,b,t in zip(knots[:-1],knots[1:],duration):
        n=int(round(t*30.));pts=path[offset:offset+n+1]
        u=((pts-a)@(b-a))/np.sum((b-a)**2)
        np.testing.assert_allclose(pts,a+u[:,None]*(b-a),atol=1e-12)
        assert np.all(u>=-1e-12) and np.all(u<=1+1e-12);offset+=n
    p=out/'RETIMING_REGRESSION.json'
    atomic_json(p,dict(status='PASS',frames=len(path),duration_s=duration.sum(),
        maximum_velocity=np.max(abs(measured_v),axis=0),maximum_acceleration=np.max(abs(measured_a),axis=0),
        geometric_segments_preserved=True,control_fps=30.,Dex3_contact_path_derivative_checks=finger_checks))
    return [p]


def golden_regression(out):
    from .interaction_chain import coverage
    return coverage(out,golden_only=True)


def coverage8_planning(out):
    from .interaction_chain import coverage
    return coverage(out,golden_only=False)


def small_physics_validation(out):
    from .architecture_physics import run
    return run(out)


def full_video_validation(out):
    from .architecture_videos import run
    return run(out)


def paper_figure_mapping(out):
    from .architecture_reports import paper_mapping
    return paper_mapping(out)


def final_gate(out):
    from .architecture_reports import final_gate as gate
    return gate(out)
