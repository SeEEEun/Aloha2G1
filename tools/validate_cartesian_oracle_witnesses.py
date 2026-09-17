#!/usr/bin/env python3
"""Independent FK, joint-limit and frozen-geometry checks of dense witnesses."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cartesian_reachability_forensic import prepare,model,DEST,BASELINE
from tools.final_single_variable_prepare import read,file_record
from tools.run_reference_motion_scientific_reset import atomic_json


def main():
    contract=prepare();g1,collision,_=model();rows=[]
    path=DEST/'INDEPENDENT_REACHABLE_WITNESS_GEOMETRY_CHECK.json'
    for case in contract['cases']:
        name=case['name']
        with np.load(BASELINE/(name+'.npz'),allow_pickle=False) as source:
            for first,last in case['failing_intervals']:
                for frame in range(first,last+1):
                    entry_path=DEST/'dense'/f'{name}_F{frame:04d}.json'
                    entry=read(entry_path)
                    if entry['classification']!='FRAME_REACHABLE':continue
                    q=np.array(entry['best']['q']);target=source['RAW_REPRESENTATION_TARGET'][frame]
                    np.testing.assert_array_equal(entry['incoming_target_m'],target)
                    assert np.isfinite(q).all()
                    margin=float(np.min(np.minimum(q-g1.arm_limits[:,0],g1.arm_limits[:,1]-q)))
                    assert margin>=0
                    g1.assign(q,*source['common_hand_q'][frame])
                    actual=np.array([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
                    residual=np.linalg.norm(actual-target,axis=1)
                    assert residual.max()<=.01
                    records=collision.inspect(q,*source['common_hand_q'][frame])
                    assert not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in records)
                    rows.append({'case':name,'frame':frame,'maximum_residual_m':float(residual.max()),
                                 'joint_limit_margin_rad':margin,'detailed_collision_records':records,
                                 'source':file_record(entry_path),'status':'PASS'})
        atomic_json(path,{'complete':False,'rows':rows,'validator':file_record(Path(__file__))})
        print('INDEPENDENT_WITNESS_GEOMETRY',name,'cumulative_pass',len(rows),flush=True)
    atomic_json(path,{'complete':True,'count':len(rows),'status':'PASS','rows':rows,
                      'validator':file_record(Path(__file__))})
    prepare()
    print('ALL_REACHABLE_WITNESSES_INDEPENDENTLY_RECONFIRMED',len(rows),flush=True)


if __name__=='__main__':main()
