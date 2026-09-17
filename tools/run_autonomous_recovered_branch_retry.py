#!/usr/bin/env python3
"""Reapply every strict gate to newly found branch seeds; no rule change."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_fixed_anchor_precision import restore

def run(case):
    verified_oracle_contract();g,c,natural=model();s=CommonPositionSolver(g,c,read(QUAL),natural);g.assign(natural);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)))
    folder=RUN/case/'recovered_branch_strict_retry_v1'
    init=RUN/case/'aggregate_step_semantics_diagnostic_v1/ATTEMPT_0.npz';q=np.load(init)['q'].copy()
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    atomic_json(folder/'CONTRACT.json',dict(source=file_record(init),maximum_attempts=2,maximum_evaluations=400,
        reason='New branch witness is a seed only. Reapply original fixed aggregate cap as well as all physical gates; release numerical certified-q anchors but retain unchanged Cartesian closest-feasible allowances.',
        acceptance_rules_unchanged=True,raw_targets_unchanged=True,implementation=file_record(Path(__file__))))
    for attempt in range(2):
        p=folder/f'ATTEMPT_{attempt}.npz';mp=folder/f'ATTEMPT_{attempt}.json'
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        if p.exists():q=np.load(p)['q'].copy();fit=read(mp)['fit']
        else:
            start=time.monotonic();print('STRICT_BRANCH_START',case,attempt,flush=True)
            q,fit=restore(s,t,h,q,dt,allow,max_nfev=400)
            fit['runtime_s']=time.monotonic()-start
            atomic_npz(p,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        atomic_json(mp,dict(metrics=met,fit=fit,trajectory=file_record(p)))
        print('STRICT_BRANCH_RESULT',case,attempt,'failures',met['failed_frames'],met['temporal'],flush=True)
        if met['pass_numeric']:
            atomic_json(folder/'NUMERIC_PASS.json',dict(metrics=met,trajectory=file_record(p),next='DETAILED_COLLISION_QUALIFICATION'));return
    atomic_json(folder/'COMPLETE.json',dict(metrics=met,global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
