#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_fixed_anchor_precision import restore as fixed_restore


def run(case):
    verified_oracle_contract();g,c,natural=model();s=CommonPositionSolver(g,c,read(QUAL),natural);g.assign(natural);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)))
    folder=RUN/case/'certified_anchor_precision_v2';slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    init=RUN/case/'certified_anchor_recovery/ATTEMPT_2.npz';q=np.load(init)['q'].copy()
    ap=RUN/case/'certified_anchor_recovery/CERTIFIED_ANCHORS.npz';a=np.load(ap);anchors=a['q'].copy();mask=a['fixed_mask'].copy()
    met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
    atomic_json(folder/'CONTRACT.json',dict(source=file_record(init),anchors=file_record(ap),
        reason='Earlier bounded solves were still reducing cost and terminated at evaluation limit. Improve inner sparse linear accuracy, not scientific acceptance.',
        maximum_attempts=3,maximum_outer_evaluations=400,maximum_LSMR_iterations=1000,LSMR_atol=1e-10,LSMR_btol=1e-10,
        raw_targets_and_all_acceptance_rules_unchanged=True,implementation=file_record(ROOT/'tools/common_fixed_anchor_precision.py')))
    for attempt in range(3):
        p=folder/f'ATTEMPT_{attempt}.npz';mp=folder/f'ATTEMPT_{attempt}.json'
        if p.exists():q=np.load(p)['q'].copy()
        else:
            print('PRECISION_RETRY_START',case,attempt,flush=True);start=time.monotonic()
            q,fit=fixed_restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=anchors)
            atomic_npz(p,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
            atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start))
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        row=read(mp);row['metrics']=met;atomic_json(mp,row)
        print('PRECISION_RETRY_RESULT',case,attempt,'failures',len(met['failed_frames']),met['temporal'],row['fit'],flush=True)
        if met['pass_numeric']:
            atomic_json(folder/'NUMERIC_PASS.json',dict(trajectory=file_record(p),metrics=met,next='common wrist-nullspace detailed geometry verification'))
            return
    atomic_json(folder/'BOUNDED_SEARCH_COMPLETE.json',dict(metrics=met,global_impossibility_proven=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
