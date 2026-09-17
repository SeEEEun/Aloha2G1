"""Resumable source-guided rebuild stages; calibration is never a stage."""
from pathlib import Path
import subprocess
import numpy as np
from .io import ROOT,OFFLINE,read,record,atomic_json,atomic_text


def source_prior(out):
    from .source_motion_prior import extract,INTERVALS
    from .interaction_candidates import acquisition_bank
    from .runtime_hulls import object_dimensions
    from .path_quality import QualityPolicy
    from dataclasses import asdict
    fixed=read(out/'COVERAGE8.json');rows=[]
    for sid in [fixed['golden'],*fixed['source_ids']]:
        phase=read(out/'source_phase'/sid/'PHASE_RECORD.json');priors={}
        for name in INTERVALS:
            priors[name]=extract(out/'source_phase'/sid,name,['left','right'])
            assert priors[name]['anchor_count']==7 and not priors[name]['hard_tracking']
        candidates=acquisition_bank(phase,read(out/'target_repair/CONTACT_CALIBRATION.json'),object_dimensions(out))
        assert len({np.asarray(c['task_space_target']).round(9).tobytes() for c in candidates})>1
        path=out/'compact_source_motion'/sid/'PRIORS.json';atomic_json(path,priors)
        rows.append(dict(source_id=sid,priors=record(path),meaningful_grasp_candidates=len(candidates)))
    path=out/'COMPACT_SOURCE_MOTION_PRIORS.json';atomic_json(path,dict(status='PASS',rows=rows,weights=asdict(QualityPolicy()),
        A_primary='WRIST_REFERENCE',B_C_primary='INTERACTION_PHASE_GOALS',B_C_source_motion='SOFT',calibration_started=False))
    return [path,*[Path(r['priors']['path']) for r in rows]]


def structural_tests(out):
    log=out/'SOURCE_GUIDED_STRUCTURAL_TESTS.log'
    with log.open('w') as stream:
        result=subprocess.run([OFFLINE,'-B','-m','unittest','tools.contact_coordination.tests.test_source_guided_rrt',
            'tools.contact_coordination.tests.test_source_guided_interaction',
            'tools.contact_coordination.tests.test_source_guided_wrist_contract',
            'tools.contact_coordination.tests.test_source_guided_physics_cache',
            'tools.contact_coordination.tests.test_source_guided_execution_watchdog',
            'tools.contact_coordination.tests.test_architecture_planner','tools.contact_coordination.test_morphology_repair',
            'tools.contact_coordination.test_abc_contract','-v'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise AssertionError('Structural tests failed: '+str(log))
    from .source_guided_robot_tests import run
    geometry=run(out)
    path=out/'SOURCE_GUIDED_STRUCTURAL_TESTS.json';atomic_json(path,dict(status='PASS',tests=57,log=record(log),
        mandatory_tests=list(range(1,10)),robot_geometry=record(geometry)))
    return [path,log,geometry]


def _planning(out,golden):
    import os
    from .interaction_chain import attempt
    fixed=read(out/'COVERAGE8.json');ids=[fixed['golden']] if golden else fixed['source_ids'];rows=[]
    path=out/('GOLDEN_PLANNING.json' if golden else 'COVERAGE8_PLANNING.json')
    for sid in ids:
        for method in ('B_INDEPENDENT','C_COUPLED'):
            rows.append(attempt(out,sid,method));atomic_json(path,dict(status='COMPLETE',rows=rows,complete=sum(r['full_task_plan'] for r in rows)))
    audit=Path(os.environ['SOURCE_GUIDED_RRT_AUDIT_LOG']) if os.environ.get('SOURCE_GUIDED_RRT_AUDIT_LOG') else None
    return [path,*[Path(r['receipt']) for r in rows],*([audit] if audit and audit.exists() else [])]


def golden_regression(out):
    from .source_guided_coverage import run
    return run(out,golden=True)
def coverage8_planning(out):
    from .source_guided_coverage import run
    return run(out)


def planning_audit(out):
    from .source_guided_reports import planning_audit
    return planning_audit(out)


def small_physics_validation(out):
    from .source_guided_physics import run
    from .pregrasp_protection import audit_all
    paths=run(out)
    return [*paths,*audit_all(out)]


def full_video_validation(out):
    from .source_guided_videos import run
    return run(out)


def final_gate(out):
    from .source_guided_reports import final_gate
    return final_gate(out)
