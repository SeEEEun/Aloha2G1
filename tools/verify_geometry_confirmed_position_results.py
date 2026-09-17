#!/usr/bin/env python3
"""Independent artifact/FK/interface validation; does not change qualification."""
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.qualify_geometry_confirmed_position import RUN,QUAL,PREVIOUS,CONTRACT,MODES,SMOKE,read,verify,file_record
from tools.run_reference_motion_scientific_reset import atomic_json
from tools.final_single_variable_prepare import status
from tools.doll_handoff_retargeting.common import load_common_config,load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics


def main():
    contract=read(CONTRACT);verify(contract)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_common_geometry_confirmed_collision*.py','-v'],cwd=ROOT,text=True,capture_output=True)
    assert test.returncode==0,test.stderr
    common=load_common_config(ROOT/'outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json')
    g1=G1Kinematics(common,load_scene(common));rows=[]
    for mode in MODES:
        for ep in SMOKE:
            name=f'{mode}_EP{ep:03d}'
            result=read(RUN/(name+'.json'))
            frames=read(RUN/(name+'_COLLISION_FRAMES.json'))
            with np.load(RUN/(name+'.npz'),allow_pickle=False) as z,np.load(PREVIOUS/(name+'.npz'),allow_pickle=False) as old:
                for key in ('RAW_REPRESENTATION_TARGET','source_timestamp','common_hand_q','initial_q'):
                    assert np.array_equal(z[key],old[key]),(name,key)
                q=z['q'];actual=[]
                assert len(frames)==len(q)
                assert [f['frame'] for f in frames]==list(range(len(q)))
                assert np.isfinite(q).all()
                for value,hands in zip(q,z['common_hand_q']):
                    g1.assign(value,*hands)
                    actual.append([g1.data.xpos[g1.wrist_ids[side]].copy() for side in ('left','right')])
                actual=np.asarray(actual)
                error=np.linalg.norm(actual-z['RAW_REPRESENTATION_TARGET'],axis=2)
                np.testing.assert_allclose(error,z['position_residual_m'],rtol=0,atol=1e-12)
                failed=np.flatnonzero(error.max(axis=1)>contract['position_acceptance']['tolerance_m']).tolist()
                assert failed==result['failing_cartesian_frames']
                count={c:sum(any(r['classification']==c for r in f['records']) for f in frames) for c in ('PROXY_ONLY_OVERLAP','HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')}
                assert count==result['classification_counts']
                assert all(f['classification']=='CLEAR' for f in frames if not f['proxy_collision'])
                for f in frames:
                    if f['classification']=='PROXY_ONLY_OVERLAP':
                        assert f['detailed_collision'] is False
                        assert f['detailed_separation_mm']>1000*contract['numerical_tolerance_m']
                limits=np.count_nonzero((q<g1.arm_limits[:,0]-1e-9)|(q>g1.arm_limits[:,1]+1e-9))
                assert limits==result['hard_limit_violations']
                rows.append({'representation_mode':mode,'episode':ep,'pass':True,'frames':len(q),'collision_counts':count,
                             'target_hand_timestamp_initial_state_exact_parity':True,'independent_fk_residual_maximum_difference_m':float(np.max(np.abs(error-z['position_residual_m'])))})
    verify(contract)
    report={'status':'PASS','rows':rows,'protected_artifact_mismatches':[],
      'unit_and_property_tests':{'returncode':test.returncode,'stdout':test.stdout,'stderr':test.stderr},
      'verification_source':file_record(Path(__file__)),
      'test_sources':[file_record(p) for p in sorted((ROOT/'tests').glob('test_common_geometry_confirmed_collision*.py'))],
      'legacy_metric_note':'self_collision_minimum_margin_m in trajectory JSON is the unchanged legacy proxy-distance diagnostic, not a hard-geometry margin. Geometry classification/counts and detailed separation are stored separately.',
      'qualification_changed':False}
    atomic_json(RUN/'INDEPENDENT_RESULT_VALIDATION.json',report)
    summary=read(QUAL/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.json')
    next_gate='COMMON_ORIENTATION_CONVENTION_AUDIT; INDEPENDENT_VALIDATION_PASS' if summary['orientation_next_gate_permitted'] else 'STOP_CARTESIAN_GATE_FAILED; INDEPENDENT_VALIDATION_PASS; NO_ORIENTATION_OR_DEX3'
    status(summary['status'],next_gate,
           [CONTRACT,QUAL/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.json',QUAL/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.md',
            QUAL/'POSITION_IK_AFTER_COLLISION_RULE.md',QUAL.parent/'FINAL_COMMON_PIPELINE_PARITY_AUDIT.md',
            RUN/'COMPLETION_HASH_MANIFEST.json',RUN/'INDEPENDENT_RESULT_VALIDATION.json'])
    print('INDEPENDENT_RESULT_VALIDATION_PASS',len(rows),'trajectories; 15 tests; protected artifacts unchanged',flush=True)


if __name__=='__main__':main()
