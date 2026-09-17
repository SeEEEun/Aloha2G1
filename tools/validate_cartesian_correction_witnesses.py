#!/usr/bin/env python3
"""Recheck physically valid upper-bound witnesses; never alter a target."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cartesian_reachability_forensic import prepare,model,DEST,BASELINE
from tools.final_single_variable_prepare import read,file_record
from tools.run_reference_motion_scientific_reset import atomic_json


def main():
    contract=prepare();g1,collision,_=model();rows=[]
    output=DEST/'INDEPENDENT_CORRECTION_WITNESS_GEOMETRY_CHECK.json'
    for case in contract['cases']:
        name=case['name'];baseline=BASELINE/(name+'.npz')
        with np.load(baseline,allow_pickle=False) as source:
            for first,last in case['failing_intervals']:
                for frame in range(first,last+1):
                    entry_path=DEST/'dense'/f'{name}_F{frame:04d}.json';entry=read(entry_path)
                    if entry['classification']!='FRAME_UNREACHABLE':continue
                    oracle_upper=entry['correction_to_satisfy_gate_upper_bound_m']
                    baseline_upper=max(float(source['position_residual_m'][frame].max())-.01,0.)
                    use_baseline=oracle_upper is None or baseline_upper<oracle_upper
                    upper=baseline_upper if use_baseline else oracle_upper
                    q=source['q'][frame] if use_baseline else np.array(entry['best']['q'])
                    target=source['RAW_REPRESENTATION_TARGET'][frame]
                    np.testing.assert_array_equal(entry['incoming_target_m'],target)
                    assert np.isfinite(q).all()
                    margin=float(np.min(np.minimum(q-g1.arm_limits[:,0],g1.arm_limits[:,1]-q)))
                    assert margin>=-1e-9
                    g1.assign(q,*source['common_hand_q'][frame])
                    actual=np.array([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
                    error=np.linalg.norm(actual-target,axis=1)
                    np.testing.assert_allclose(max(error.max()-.01,0.),upper,rtol=0,atol=1e-11)
                    assert entry['correction_to_gate_certified_lower_bound_m']<=upper+1e-8
                    records=collision.inspect(q,*source['common_hand_q'][frame])
                    assert not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in records)
                    rows.append({'case':name,'frame':frame,'q':q.tolist(),'joint_limit_margin_rad':margin,
                        'correction_to_gate_upper_bound_m':upper,'detailed_collision_records':records,
                        'witness_source':'FROZEN_COLLISION_VALID_BASELINE' if use_baseline else 'COLLISION_VALID_ORACLE_CANDIDATE',
                        'source':file_record(entry_path),'baseline':file_record(baseline),'status':'PASS'})
        atomic_json(output,{'complete':False,'rows':rows,'validator':file_record(Path(__file__))})
        print('INDEPENDENT_CORRECTION_GEOMETRY',name,'cumulative_pass',len(rows),flush=True)
    atomic_json(output,{'complete':True,'count':len(rows),'status':'PASS','rows':rows,
                        'validator':file_record(Path(__file__))})
    prepare()
    print('ALL_CORRECTION_WITNESSES_INDEPENDENTLY_RECONFIRMED',len(rows),flush=True)


if __name__=='__main__':main()
