#!/usr/bin/env python3
"""Detailed geometry evidence for non-promoted temporal-semantics diagnostics."""
from pathlib import Path
import sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *

def run(case):
    verified_oracle_contract();g,c,n=model();s=CommonPositionSolver(g,c,read(QUAL),n)
    folder=RUN/case/'aggregate_step_semantics_diagnostic_v1'
    src=folder/'ATTEMPT_0.npz';z=np.load(src);q=z['q'];h=z['common_hand_q']
    rankpath=folder/'GEOMETRY_SEED_RANKING.json'
    if rankpath.exists():rank=read(rankpath)['rows']
    else:
        rank=rank_constant_nullspace(g,c,q,h)
        atomic_json(rankpath,dict(source=file_record(src),rows=rank,implementation=file_record(ROOT/'tools/common_wrist_nullspace_search.py')))
    origin=np.array([s.pose_jacobian(v,hh)[0] for v,hh in zip(q,h)])
    for k,row in enumerate(rank[:4]):
        p=folder/f'DETAILED_GEOMETRY_{k}.json';qp=folder/f'DETAILED_GEOMETRY_{k}.npz'
        if p.exists():
            if not read(p)['blocked_frames']:break
            continue
        v=q.copy();v[:,[6,13]]=row['wrist_joint_values']
        actual=np.array([s.pose_jacobian(vv,hh)[0] for vv,hh in zip(v,h)])
        np.testing.assert_allclose(actual,origin,rtol=0,atol=1e-12)
        records=[]
        for f,(vv,hh) in enumerate(zip(v,h)):
            records.append(dict(frame=f,records=c.inspect(vv,*hh)))
            if f%150==0:print('STEP_DIAGNOSTIC_GEOMETRY',case,k,f,flush=True)
        blocked=[r['frame'] for r in records if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        atomic_npz(qp,q=v,RAW_REPRESENTATION_TARGET=z['RAW_REPRESENTATION_TARGET'],common_hand_q=h,source_timestamp=z['source_timestamp'])
        atomic_json(p,dict(status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',source=file_record(src),trajectory=file_record(qp),
            geometry=records,blocked_frames=blocked,seed=row,position_preservation_exact=True,
            changed_joints='constant terminal wrist hinges only; position-nullspace; common49-seed rule'))
        print('STEP_DIAGNOSTIC_GEOMETRY_DONE',case,k,'blocked',len(blocked),flush=True)
        if not blocked:break
    atomic_json(folder/'GEOMETRY_COMPLETE.json',dict(status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',last_report=file_record(p)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
