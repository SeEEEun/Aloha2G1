"""Small shared transform, source, cache, recording and gate regressions."""
import subprocess
import os
from .io import ROOT,OFFLINE,record,atomic_json


def source_checks(out):
    import copy
    import numpy as np
    from .io import read
    from .scientific_cache import scientific_inputs,digest
    from .receiving_relation import build_region
    from .morphology_repair import wrist_target
    from .handoff_repair import coupling_residual
    from .runtime_hulls import object_dimensions
    from .source_phase import mean_pose
    ids=read(out/'SPLIT_CONTRACT.json')['authorized_training_source_ids']
    selected=[ids[i] for i in [0,10,20,29,39]]
    calibration=read(out/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json')
    base=read(out/'target_repair/CONTACT_CALIBRATION.json');separation=read(out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json')['offset_object_frame_m']
    gravity=calibration['receiver_capture_transition']['gravity_up_in_preobject'];rows=[]
    for sid in selected:
        phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
        contact=calibration['contacts']['receiver_acquisition_intent']
        a=build_region(phase,contact,object_dimensions(out),separation,gravity)
        p=copy.deepcopy(phase);p['source_functional_tool_object_relations']['right'][0][3]+=.002
        b=build_region(p,contact,object_dimensions(out),separation,gravity)
        delta=float(np.max(np.abs(a['target_T_H_O']-b['target_T_H_O'])))
        assert delta>0 and p['source_functional_tool_object_relations']['left']==phase['source_functional_tool_object_relations']['left']
        x=np.asarray(phase['initial_object_pose_world']);acq=wrist_target(x,base['contacts']['acquisition_intent'])
        moved=x.copy();moved[0,3]+=.02
        np.testing.assert_allclose(wrist_target(moved,base['contacts']['acquisition_intent'])[:3,3]-acq[:3,3],[.02,0,0],atol=1e-12)
        inputs=scientific_inputs(out,sid);before=digest(inputs)
        changed=copy.deepcopy(inputs);changed['source']['source_functional_tool_object_relations']['right'][0][3]+=.002
        assert digest(changed)!=before
        changed=copy.deepcopy(inputs);changed['source']['initial_object_pose_world'][0][3]+=.02
        assert digest(changed)!=before
        changed=copy.deepcopy(inputs);changed['calibrated_parameters']['handoff_posture_multiplier']=.5
        assert digest(changed)!=before
        changed=copy.deepcopy(inputs);changed['plotting']={'dpi':900}
        assert digest(changed)==before
        prior=np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz')
        handoff=mean_pose(prior['inferred_object_from_left'][phase['handoff_sample_indices']])
        rows.append(dict(source_id=sid,acquisition_world=acq,source_left_relation=phase['source_functional_tool_object_relations']['left'],
            source_receiving_relation=phase['source_functional_tool_object_relations']['right'],receiving_region=a,
            source_handoff_prior=handoff,receiving_mutation_delta=delta,scientific_input_digest=before,
            cache_relation_object_parameter_changes_miss=True,plot_metadata_does_not_invalidate=True))
    left=np.eye(4);right=np.eye(4);right[0,3]=.005
    assert np.linalg.norm(coupling_residual(left,right,True))>0
    np.testing.assert_array_equal(coupling_residual(left,right,False),np.zeros(6))
    result=dict(status='PASS',sources=rows,source_id_branch=False,mutations_persisted_as_experiments=False,
        full_q_trajectory_reused=False,coupling_algebra_nonredundant=True,
        limits='Acquisition is an object-conditioned target calibration family; exact source TCP contact axes are not reproduced. Handoff source object prior is inferred from the left relation. Physical semantic validity remains required.')
    atomic_json(out/'SOURCE_AND_CACHE_REGRESSION.json',result);return result


def run(out,resume=False):
    tests=['tools.contact_coordination.test_abc_contract',
           'tools.contact_coordination.test_receiving_relation',
           'tools.contact_coordination.tests.test_dev_task_frame',
           'tools.contact_coordination.tests.test_contact_candidate_region',
           'tools.contact_coordination.test_morphology_repair']
    log=out/'CORE_REGRESSION.log'
    with log.open('w') as stream:
        result=subprocess.run([OFFLINE,'-B','-m','unittest',*tests],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    report=dict(status='PASS' if result.returncode==0 else 'FAIL',returncode=result.returncode,
                tests=tests,log=record(log),science_physics_executed=False)
    if result.returncode==0:
        source_checks(out);report['source_and_cache']=record(out/'SOURCE_AND_CACHE_REGRESSION.json')
    atomic_json(out/'CORE_REGRESSION.json',report);return report
