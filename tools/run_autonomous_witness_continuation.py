#!/usr/bin/env python3
"""Common independent-frame anchors followed by whole-trajectory continuation.

Only the orchestration layer selects frozen files. Numerical inputs contain
positions, hand articulation, time and actual model constraints, never labels.
"""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_position_candidates import kinematic_metrics
from tools.run_master_autonomous import MASTER,RUN,QUAL,read,file_record,atomic_json,atomic_npz
from tools.cartesian_reachability_forensic import model,BASELINE,DEST
from tools.master_autonomous_provenance import verified_oracle_contract
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.final_common_position_solver import CommonPositionSolver
from tools.common_g1_position_bounds import outer_enclosures
from tools.master_autonomous_common import refine_trajectory


def independent_anchors(oracle,targets,hands,seed):
    q=[]
    for target,hand in zip(targets,hands):
        value,_=oracle.fit(target,hand,seed,max_evaluations=180)
        q.append(value)
    return np.array(q)


def run(case):
    cfg=read(QUAL);oc=verified_oracle_contract()[0];g1,collision,natural=model()
    oracle=FramewiseReachabilityOracle(g1,collision,oc['oracle'],natural)
    solver=CommonPositionSolver(g1,collision,cfg,natural);g1.assign(natural);balls=outer_enclosures(g1)
    source=BASELINE/f'{case}.npz'
    with np.load(source) as z:target=z['RAW_REPRESENTATION_TARGET'].copy();hands=z['common_hand_q'].copy();ts=z['source_timestamp'].copy();baseline=z['q'].copy()
    dt=float(np.median(np.diff(ts)));folder=RUN/case/'witness_continuation';rows=[]
    path=folder/'INDEPENDENT_NATURAL.npz'
    if path.exists():q0=np.load(path)['q'].copy()
    else:
        q0=independent_anchors(oracle,target,hands,natural)
        atomic_npz(path,q=q0,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=ts)
    # Reuse geometry-verified dense witnesses at their exact target/hand state.
    # They are initial guesses only. Unknown frames use the same natural seed.
    witnessed=q0.copy()
    for f in range(len(target)):
        p=DEST/'dense'/f'{case}_F{f:04d}.json'
        if not p.exists():continue
        record=read(p);np.testing.assert_array_equal(record['incoming_target_m'],target[f])
        np.testing.assert_array_equal(record['common_hand_q'],hands[f])
        if record['best']['geometry_valid']:witnessed[f]=record['best']['q']
    for label,initial in [('independent',q0),('verified_witness',witnessed)]:
        for weight in (.03,.01,.003):
            tag=f'{label}_{weight}';path=folder/f'{tag}.npz';mp=folder/f'{tag}.json'
            if path.exists():q=np.load(path)['q'].copy();fit=read(mp)['fit']
            else:
                print('WITNESS_REFINE_START',case,tag,flush=True);t=time.monotonic()
                q,fit=refine_trajectory(solver,target,hands,initial,weight,100);fit['runtime_s']=time.monotonic()-t
                atomic_npz(path,q=q,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=ts)
            metrics,actual,residual,lower=kinematic_metrics(solver,q,target,hands,dt,cfg,balls)
            atomic_json(mp,dict(metrics=metrics,fit=fit,trajectory=file_record(path),implementation=file_record(Path(__file__))))
            print('WITNESS_RESULT',case,tag,'failures',len(metrics['screen_failures']),metrics['temporal'],flush=True)
            rows.append((tag,q,metrics))
    candidates=sorted(rows,key=lambda r:(len(r[2]['screen_failures']),not r[2]['temporal']['pass_temporal'],r[2]['temporal']['maximum_step_norm_rad']))
    for label,q,metrics in candidates[:2]:
        geometry=[]
        for f,(v,h) in enumerate(zip(q,hands)):
            records=collision.inspect(v,*h);geometry.append(dict(frame=f,records=records))
            if f%100==0:print('WITNESS_GEOMETRY',case,label,f,flush=True)
        blocked=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        atomic_json(folder/f'{label}_GEOMETRY.json',dict(blocked_frames=blocked,geometry=geometry,metrics=metrics))
        print('WITNESS_GEOMETRY_COMPLETE',case,label,len(blocked),flush=True)
    atomic_json(folder/'COMPLETE.json',dict(candidates=[dict(label=a,metrics=c) for a,b,c in rows],qualified=False,
        note='No promotion without complete physical, source-join and raw/closest-feasible qualification'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
