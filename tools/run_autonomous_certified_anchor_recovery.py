#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_fixed_anchor_continuation import restore as fixed_restore


def run(case):
    oc=verified_oracle_contract()[0];g,c,natural=model();s=CommonPositionSolver(g,c,read(QUAL),natural);g.assign(natural);bounds=orbit_enclosures(g)
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],natural)
    t,h,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)))
    folder=RUN/case/'certified_anchor_recovery';slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    init=RUN/case/'dual_position_unsmoothed_anchors_v1/RESTORED_2.npz';q=np.load(init)['q'].copy()
    met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
    mask=np.zeros_like(q,dtype=bool);anchors=q.copy();ap=folder/'CERTIFIED_ANCHORS.npz'
    if ap.exists():
        z=np.load(ap);anchors=z['q'].copy();mask=z['fixed_mask'].copy()
    else:
        previous=natural.copy()
        for f in np.flatnonzero(cert):
            fitted,info=oracle.fit(t[f],h[f],previous,max_evaluations=360)
            for arm in range(2):
                if lb[f,arm]>.01000001:
                    assert info['residual_m'][arm]<=allow[f,arm],(f,arm,info)
                    anchors[f,arm*7:arm*7+6]=fitted[arm*7:arm*7+6];mask[f,arm*7:arm*7+6]=True
            previous=fitted
        atomic_npz(ap,q=anchors,fixed_mask=mask,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    q[mask]=anchors[mask]
    atomic_json(folder/'CONTRACT.json',dict(source=source,anchor=file_record(ap),
        rule='Exact collision-unqualified closest-position witness coordinates fixed only as a bounded candidate family; other coordinates retain whole-trajectory constraints. Terminal wrist hinges remain free. No raw target changes.',
        maximum_attempts=3,maximum_evaluations_per_attempt=240,implementation=file_record(ROOT/'tools/common_fixed_anchor_continuation.py')))
    for attempt in range(3):
        p=folder/f'ATTEMPT_{attempt}.npz';mp=folder/f'ATTEMPT_{attempt}.json'
        if p.exists():q=np.load(p)['q'].copy()
        else:
            print('FIXED_ANCHOR_START',case,attempt,flush=True);start=time.monotonic()
            q,fit=fixed_restore(s,t,h,q,dt,allow,max_nfev=240,fixed_mask=mask,fixed_values=anchors)
            atomic_npz(p,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
            atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start))
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        row=read(mp);row['metrics']=met;atomic_json(mp,row)
        print('FIXED_ANCHOR_RESULT',case,attempt,'failures',len(met['failed_frames']),met['temporal'],flush=True)
        if met['pass_numeric']:
            atomic_json(folder/'NUMERIC_PASS.json',dict(trajectory=file_record(p),metrics=met,next='common wrist-nullspace and detailed collision verification'))
            return
    atomic_json(folder/'BOUNDED_SEARCH_COMPLETE.json',dict(metrics=met,global_impossibility_proven=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
