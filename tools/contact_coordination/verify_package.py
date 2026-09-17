#!/usr/bin/env python3
"""Verify prototype accounting, measured evidence and reproducible dependency hashes."""
import argparse
import csv
import datetime
import io
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_coordination.io import read,record,atomic_json,atomic_text


def verify(out):
    from tools.contact_coordination.test_hybrid import HybridTests
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HybridTests))
    atomic_text(out/'validation/UNIT_TESTS.txt',stream.getvalue())
    assert tests.wasSuccessful()
    selection=read(out/'bootstrap/SELECTION.json')
    rows=list(csv.DictReader((out/'PER_INSTANCE_RESULTS.csv').open()))
    assert len(rows)==80
    for method,n in [('WRIST_REFERENCE',35),('INTERACTION_OURS',35),('OURS_NO_COUPLING',10)]:
        subset=[r for r in rows if r['method']==method]
        assert len(subset)==n and len({r['source_id'] for r in subset})==n
        assert all(r['terminal_class']=='NOT_ATTEMPTED_UPSTREAM' and r['physically_run']=='False' for r in subset)
    assert [int(r['dev_position']) for r in rows if r['method']=='OURS_NO_COUPLING']==selection['ablation_dev_positions']
    split=read(out/'bootstrap/SPLITS.json')['entries'];train={r['source_recording_id'] for r in split if r['TRAIN40']}
    assert set(selection['train_source_ids'])<=train
    for entry in split:
        assert record(entry['source_parquet'])['sha256']==entry['source_parquet_sha256']
    for p in (out/'source_phase').glob('*/PHASE_RECORD.json'):
        phase=read(p)
        for name in ('raw','reference','image_registration','physics'):
            assert record(phase['provenance'][name]['path'])==phase['provenance'][name]
    trial=read(out/'common_control/scripted_captured/trial_result.json')
    assert trial['object_pose_writes_during_timed_loop']==0 and not trial['prohibited_attachment_used']
    assert not trial['state_restoration']['used']
    audit=read(out/'common_control/scripted_captured/RAW_NUMERICAL_AND_RETENTION_AUDIT.json')
    assert record(audit['trace']['path'])==audit['trace']
    assert audit['maximum_raw_joint_bound_excess_rad']==0.
    replay=read(out/'replays/COMMON_CONTROL_REPLAY.json')
    assert record(replay['video']['path'])==replay['video']
    assert replay['trace']==audit['trace'] and not replay['method_result']
    subprocess.run(['ffmpeg','-v','error','-i',replay['video']['path'],'-f','null','-'],check=True,timeout=60)
    # Preserve the original tracked patch exactly; no source/result deletion.
    initial=(out/'bootstrap/tracked_changes_initial.patch').read_bytes()
    assert subprocess.check_output(['git','diff','--binary'],cwd=ROOT)==initial
    for p in (out/'bootstrap/preexisting_code').glob('*.py'):
        assert p.read_bytes()==(ROOT/'tools/contact_coordination'/p.name).read_bytes()
    assets=[]
    common=read(ROOT/'outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json')
    for key,path in common['models'].items():
        if key.endswith('_sha256'):continue
        rec=record(path)
        if key+'_sha256' in common['models']:assert rec['sha256']==common['models'][key+'_sha256']
        assets.append(rec)
    geometry=read(out/'common_control/scripted_captured/RUNTIME_COLLISION_MODEL.json')
    for path in geometry['usd_layers']:
        if Path(path).is_file():assets.append(record(path))
    model_root=Path(common['models']['g1_xml']).parent
    for p in sorted(model_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.xml','.stl','.obj','.ply'):
            assets.append(record(p))
    atomic_json(out/'reproduction/MODEL_AND_RUNTIME_ASSETS.json',assets)
    for directory in ['tools/contact_coordination','configs/contact_coordination']:
        for path in (ROOT/directory).iterdir():
            if path.is_file() and path.suffix in ('.py','.json'):
                destination=out/'reproduction/code'/path.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,destination)
    atomic_text(out/'REPRODUCE.md',f'''# Reproduce the supported prototype package

Working directory: `{ROOT}`. No ACT, VLA, hardware or DEV physical execution is included.

```
/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python -m unittest tools.contact_coordination.test_hybrid -v
/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python tools/contact_coordination/run_hybrid_study.py --run-dir {out} --stage prototype --resume
/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python tools/contact_coordination/run_hybrid_study.py --run-dir {out} --stage report --resume
MUJOCO_GL=egl /home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python tools/contact_coordination/run_hybrid_study.py --run-dir {out} --stage render --resume
/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python tools/contact_coordination/verify_package.py --run-dir {out}
```

The run lock prevents simultaneous stages. Resume checks hashes and preserves failed scientific candidates. Changed development kernels require a new run directory. `primary_dev35` returns NOT_ATTEMPTED_UPSTREAM while M2 is absent. Later full-study stages have not been implemented/qualified in this prototype.

Code snapshot: `reproduction/code/`. Asset versions: `reproduction/MODEL_AND_RUNTIME_ASSETS.json`. Raw source identities/versions: `bootstrap/SPLITS.json`. Existing tracked modifications before this work: `bootstrap/tracked_changes_initial.patch`; pre-existing helper files are also checkpointed. Runtime physics invocation and all measured states: `common_control/scripted_captured/`.

`--stage common_control --resume` reaudits exact compatible controls and does not rerun a completed physical attempt. For a new run it launches one existing calibration command using the explicit read-only capture wrapper, at most once in that output directory. Interrupted attempts are retained; no endless automatic retry is implemented.

The 80 DEV rows are planned nonexecution accounting. Do not interpret these files or the scripted calibration video as a completed physical comparison.
''')
    result=dict(status='PASS',unit_tests_run=tests.testsRun,unit_test_failures=len(tests.failures),unit_test_errors=len(tests.errors),
        intended_dev_instances=len(rows),physical_method_rollouts=0,source_conditioned_full_task='NOT_DEMONSTRATED',
        raw_recordings_hash_verified=len(split),code_snapshot=True,prior_tracked_changes_unchanged=True,
        prior_contact_coordination_helpers_unchanged=True,common_control_video_decoded=True,
        actual_measured_common_control=True,study_freeze_exists=False,
        terminal_status='HYBRID_RETARGETING_PROTOTYPE_REPORT_ONLY')
    atomic_json(out/'validation/PACKAGE_VERIFICATION.json',result)
    paths=[p for p in out.rglob('*') if p.is_file() and p.name not in ('ARTIFACT_HASH_MANIFEST.json','RUN.lock') and '.incomplete' not in p.name]
    atomic_json(out/'ARTIFACT_HASH_MANIFEST.json',dict(purpose='prototype evidence inventory, not an evaluation freeze',files=[record(p) for p in sorted(paths)]))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    print(verify(args.run_dir.resolve()))
