"""Persist bounded TRAIN source-conditioned phase feasibility evidence."""
import numpy as np
from .io import ROOT, read, record, fingerprint, atomic_json, atomic_npz
from .source_phase import COMMON
from .targets import make_goals
from .planner import realize_phase_goals

CONFIG = ROOT/'configs/contact_coordination/hybrid_development_v1.json'
INITIAL = ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json'


def run(out, resume=False):
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    source_id = read(out/'bootstrap/SELECTION.json')['prototype_source_id']
    source = out/'source_phase'/source_id
    phase = read(source/'PHASE_RECORD.json')
    assert phase['split']=='TRAIN40'
    priors = dict(np.load(source/'SOURCE_PRIORS.npz',allow_pickle=False))
    config=read(CONFIG);common=load_common_config(COMMON);g1=G1Kinematics(common,load_scene(common))
    q0=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad'])
    key, files=fingerprint([CONFIG,COMMON,INITIAL,source/'PHASE_RECORD.json',source/'SOURCE_PRIORS.npz',
                           ROOT/'tools/contact_coordination/targets.py',ROOT/'tools/contact_coordination/planner.py',
                           ROOT/'tools/doll_handoff_retargeting/models.py',g1.path,g1.mapping_path])
    results=[]
    for method, representation, coupling in [('INTERACTION_OURS','INTERACTION_OURS',True),
        ('WRIST_REFERENCE','WRIST_REFERENCE',False),('OURS_NO_COUPLING','INTERACTION_OURS',False)]:
        folder=out/'prototype'/source_id/method/key[:12]
        done=folder/'RESULT.json'
        if resume and done.exists():
            old=read(done)
            assert old['dependency_key']==key
            assert record(old['targets']['path'])==old['targets']
            assert record(old['phase_ik']['path'])==old['phase_ik']
            results.append(old);continue
        targets=make_goals(phase,priors,config['target'],representation,coupling)
        atomic_json(folder/'PHASE_TARGETS.json',targets)
        atomic_json(folder/'DEPENDENCIES.json',dict(key=key,files=files))
        print('TRAIN_PHASE_IK_START',source_id,method,flush=True)
        ik=realize_phase_goals(g1,targets['phase_goals'],q0,config['planner'])
        atomic_npz(folder/'PHASE_CONFIGURATIONS.npz',q=ik.pop('q'),natural_q0=q0)
        atomic_json(folder/'PHASE_IK.json',ik)
        first=next((r for r in ik['phases'] if not r['goal_satisfied']),None)
        result=dict(source_id=source_id,method=method,dependency_key=key,
            targets=record(folder/'PHASE_TARGETS.json'),phase_ik=record(folder/'PHASE_IK.json'),
            phase_configurations=record(folder/'PHASE_CONFIGURATIONS.npz'),
            status='DEVELOPMENT_PHASE_CONSTRAINTS_UNSATISFIED' if first else 'CONNECTING_VALIDATION_REQUIRED',
            first_failure=first['phase'] if first else None,full_task_demonstrated=False,
            executable=False,physically_run=False,physical_completion='NOT_MEASURED',
            interpretation='bounded phase IK diagnostic; not a physical attempt or global impossibility proof',
            runtime_s=ik['runtime_s'])
        atomic_json(done,result);results.append(result)
        print('TRAIN_PHASE_IK_END',method,result['status'],result['first_failure'],flush=True)
    result=dict(status='M2_NOT_DEMONSTRATED',results=results,source_conditioned=True,dev35_started=False)
    atomic_json(out/'prototype/RESULT.json',result)
    return result
