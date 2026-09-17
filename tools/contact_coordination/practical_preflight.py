"""Test authorized scalar/policy extension before freezing sweep dependencies."""
from pathlib import Path
import subprocess
from .io import ROOT,OFFLINE,read,record,atomic_json,atomic_text
from .practical_study import BASE,assert_backend,stage_inputs


def run(out):
    out=Path(out).resolve();backend=assert_backend(out)
    logfile=out/'PREFLIGHT_REGRESSION.log'
    modules=['tools.contact_coordination.tests.test_practical_calibration',
        'tools.contact_coordination.tests.test_source_guided_rrt','tools.contact_coordination.tests.test_source_guided_interaction',
        'tools.contact_coordination.tests.test_source_guided_wrist_contract','tools.contact_coordination.tests.test_source_guided_physics_cache',
        'tools.contact_coordination.tests.test_source_guided_execution_watchdog','tools.contact_coordination.tests.test_architecture_planner',
        'tools.contact_coordination.test_morphology_repair','tools.contact_coordination.test_abc_contract']
    with logfile.open('w') as f:r=subprocess.run([OFFLINE,'-B','-m','unittest',*modules,'-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise AssertionError('Practical regression failed: '+str(logfile))
    area=out/'preflight_geometry';stage_inputs(BASE,area)
    atomic_json(area/'SOURCE_GUIDED_RRT_REBUILD.json',dict(old_run=str(ROOT/'outputs/converter_architecture_repair/20260910T094737Z')))
    from .source_guided_robot_tests import run as geometry
    witness=geometry(area)
    from .scientific_cache import CODE_FILES
    code=[ROOT/'tools/contact_coordination'/n for n in CODE_FILES]
    code+=[ROOT/'tools/contact_coordination'/n for n in ('pregrasp_protection.py','full_attempt.py','full_attempt_runtime.py','full_attempt_physics.py','abc_score.py','score_hybrid.py','contact_command_geometry.py')]
    original=read(BASE/'PAPER_METHOD_ALIAS.json')['workspace_code_manifest']
    code += [Path(x['path']) for x in original if not Path(x['path']).is_relative_to(ROOT/'tools/contact_coordination')]
    ids=read(out/'PRACTICAL_STUDY.json')['train_source_ids']
    inputs=[p for sid in ids for p in (out/'source_phase'/sid).iterdir() if p.suffix in ('.json','.npz')]
    inputs += [p for p in (out/'target_repair').rglob('*') if p.is_file()]
    inputs += [p for p in (out/'recipes').glob('*/PRACTICAL_PARAMETERS.json')]+[p for p in (out/'recipes').glob('*/CALIBRATION_PARAMETERS.json')]
    files=[record(p) for p in sorted(set(code+inputs))]
    atomic_json(out/'IMPLEMENTATION_FREEZE.json',dict(files=files,scope='Sweep scientific code/source/geometry/parameters; orchestration/report-only files are separate stage dependencies',
        architecture_baseline=record(out/'ARCHITECTURE_BASELINE_MANIFEST.json'),fixed_recipes=record(out/'CALIBRATION_SEARCH_SPACE.json')))
    import re
    count=int(re.search(r'Ran (\d+) tests',logfile.read_text()).group(1))
    atomic_json(out/'PREFLIGHT_REGRESSION.json',dict(status='PASS',tests=count,log=record(logfile),robot_geometry=record(witness),
        default_all40_targets_identical=True,recipes=4,incidental_policy_scoped=True,
        backend_unchanged=backend['backend_bytes_unchanged'],backend_verification=backend))
    atomic_text(out/'CURRENT_STATUS.md','CURRENT_STAGE: planning_sweep\nLAST_COMPLETED_STAGE: preflight\nCURRENT_BLOCKER: None\nNEXT_ACTION: Launch persistent planning-only sweep\nACTIVE_PROCESS: None\nDEV35_STARTED: NO\nACT_STARTED: NO\n')
    print(f'PREFLIGHT PASS: {count} tests; strict default all40 target parity; actual protected/carried geometry; '
        f"unchanged backend files={backend['unchanged_backend_files']}; verified software consistency repairs={len(backend['verified_software_consistency_repairs'])}.")


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run(a.run_dir)
