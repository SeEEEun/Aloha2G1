#!/usr/bin/env python3
from pathlib import Path
import sys,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_master_autonomous import RUN,QUAL,read,file_record,atomic_json,atomic_npz
from tools.cartesian_reachability_forensic import model
from tools.final_common_position_solver import CommonPositionSolver
from tools.common_wrist_nullspace_search import rank_constant_nullspace
from tools.master_autonomous_common import temporal_metrics


def run(case,relative):
    g1,collision,natural=model();cfg=read(QUAL);solver=CommonPositionSolver(g1,collision,cfg,natural)
    path=RUN/case/relative;folder=RUN/case/'nullspace_recovery'
    with np.load(path) as z:q=z['q'].copy();h=z['common_hand_q'].copy();target=z['RAW_REPRESENTATION_TARGET'].copy();ts=z['source_timestamp'].copy()
    origin=np.array([solver.pose_jacobian(v,hand)[0] for v,hand in zip(q,h)])
    rp=folder/'SEED_RANKING.json'
    if rp.exists():rank=read(rp)['rows']
    else:
        print('NULLSPACE_SEED_RANKING',case,flush=True)
        rank=rank_constant_nullspace(g1,collision,q,h)
        atomic_json(rp,dict(rows=rank,input=file_record(path),implementation=file_record(ROOT/'tools/common_wrist_nullspace_search.py'),
            rule='49 fixed constant terminal-hinge seeds, proxy rank only, top four full detailed confirmations; no pair exception'))
    for k,row in enumerate(rank[:4]):
        result=folder/f'CANDIDATE_{k}.json';p=folder/f'CANDIDATE_{k}.npz'
        if result.exists():continue
        candidate=q.copy();candidate[:,[6,13]]=row['wrist_joint_values']
        actual=np.array([solver.pose_jacobian(v,hand)[0] for v,hand in zip(candidate,h)])
        np.testing.assert_allclose(actual,origin,atol=1e-12,rtol=0)
        geometry=[]
        for f,(v,hand) in enumerate(zip(candidate,h)):
            rr=collision.inspect(v,*hand);geometry.append(dict(frame=f,records=rr))
            if f%100==0:print('NULLSPACE_GEOMETRY',case,k,f,flush=True)
        blocked=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        temporal=temporal_metrics(candidate,float(np.median(np.diff(ts))),cfg)
        atomic_npz(p,q=candidate,RAW_REPRESENTATION_TARGET=target,actual_wrist_position_model=actual,common_hand_q=h,source_timestamp=ts)
        atomic_json(result,dict(seed=row,geometry=geometry,blocked_frames=blocked,temporal=temporal,
            wrist_positions_unchanged=True,trajectory=file_record(p),source=file_record(path)))
        print('NULLSPACE_RESULT',case,k,'blocked',len(blocked),'temporal',temporal['pass_temporal'],flush=True)
        if not blocked and temporal['pass_temporal']:
            atomic_json(folder/'POSITION_PRESERVING_COLLISION_RECOVERY_PASS.json',dict(candidate=file_record(p),report=file_record(result),
                source_task_gate_requires_separate_validation=True,preparation_join_requires_validation=True))
            break


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('relative');a=p.parse_args();run(a.case,a.relative)
