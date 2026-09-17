#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_position_candidates import kinematic_metrics
from tools.run_master_autonomous import MASTER,RUN,QUAL,read,file_record,atomic_json,atomic_npz
from tools.cartesian_reachability_forensic import model,BASELINE
from tools.master_autonomous_provenance import verified_oracle_contract
from tools.final_common_position_solver import CommonPositionSolver
from tools.common_g1_position_bounds import outer_enclosures,residual_lower_bounds
from tools.common_constrained_continuation import restore


def run(case):
    verified_oracle_contract();cfg=read(QUAL);g1,collision,natural=model()
    solver=CommonPositionSolver(g1,collision,cfg,natural);g1.assign(natural);balls=outer_enclosures(g1)
    with np.load(BASELINE/f'{case}.npz') as z:target=z['RAW_REPRESENTATION_TARGET'].copy();hands=z['common_hand_q'].copy();ts=z['source_timestamp'].copy()
    dt=float(np.median(np.diff(ts)));folder=RUN/case/'constrained_continuation'
    init=RUN/case/'witness_continuation/independent_0.03.npz';q=np.load(init)['q'].copy()
    lower=np.array([residual_lower_bounds(t,balls) for t in target]);cert=lower.max(axis=1)>.01000001
    allowance=np.full((len(q),2),.01)
    allowance[cert]=np.maximum(lower[cert]+.002,.01) # absolute user cap, screening only
    pairs={}
    for attempt in range(3):
        p=folder/f'ATTEMPT_{attempt}.npz';mp=folder/f'ATTEMPT_{attempt}.json'
        if p.exists():q=np.load(p)['q'].copy();fit=read(mp)['fit']
        else:
            print('RESTORATION_START',case,attempt,flush=True);start=time.monotonic()
            q,fit=restore(solver,target,hands,q,dt,allowance,pairs,max_nfev=160);fit['runtime_s']=time.monotonic()-start
            atomic_npz(p,q=q,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=ts)
        metrics,actual,residual,bounds=kinematic_metrics(solver,q,target,hands,dt,cfg,balls)
        atomic_json(mp,dict(fit=fit,metrics=metrics,source=file_record(init),trajectory=file_record(p),implementation=file_record(ROOT/'tools/common_constrained_continuation.py')))
        print('RESTORATION_KINEMATICS',case,attempt,'failures',len(metrics['screen_failures']),metrics['temporal'],flush=True)
        gp=folder/f'ATTEMPT_{attempt}_GEOMETRY.json'
        if gp.exists():geometry=read(gp)['geometry']
        else:
            geometry=[]
            for f,(v,h) in enumerate(zip(q,hands)):
                rr=collision.inspect(v,*h);geometry.append(dict(frame=f,records=rr))
                if f%100==0:print('RESTORATION_GEOMETRY',case,attempt,f,flush=True)
            atomic_json(gp,dict(geometry=geometry))
        pairs={}
        for row in geometry:
            for r in row['records']:
                if r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY'):
                    pairs.setdefault(row['frame'],[]).append(tuple(r['geom_pair']))
        print('RESTORATION_COLLISION',case,attempt,'blocked',len(pairs),flush=True)
        if not metrics['screen_failures'] and metrics['temporal']['pass_temporal'] and not pairs:
            atomic_json(folder/'SOURCE_KINEMATIC_PHYSICAL_PASS.json',dict(metrics=metrics,trajectory=file_record(p),geometry=file_record(gp),
                final_qualification=False,remaining='preparation join; closest-feasible optimality where relevant; complete paired TRAIN qualification'))
            return
    atomic_json(folder/'BOUNDED_RESTORATION_COMPLETE.json',dict(kinematics=metrics,blocked_frames=sorted(pairs),
        next='Diagnose persisted failures; no automatic scientific stop after one family'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
