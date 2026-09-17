#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_trust_window_search import solve_window


def run(case):
    cfg=read(QUAL);g1,c,natural=model();s=CommonPositionSolver(g1,c,cfg,natural);g1.assign(natural);b=orbit_enclosures(g1)
    t,h,ts,source=load_input(case,g1,natural);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    qp=RUN/case/'dual_position_feasible_history_v1/RESTORED_2.npz';q=np.load(qp)['q'].copy()
    metrics,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,b,slack)
    old=read(RUN/case/'hard_window_forensic/WINDOW_SEARCH_CONTRACT.json');folder=RUN/case/'trust_window_forensic';rows=[]
    atomic_json(folder/'CONTRACT.json',dict(source=file_record(qp),selection=file_record(RUN/case/'hard_window_forensic/WINDOW_SEARCH_CONTRACT.json'),
        maximum_iterations_per_seed=350,seed_count=2,solver='trust-constr with exact sparse Jacobian and BFGS constraint Hessian',
        internal_1um_inward_margin=True,acceptance_unchanged=True,implementation=file_record(ROOT/'tools/common_trust_window_search.py')))
    for f in old['frames']:
        start=max(0,f-12);end=min(len(q),f+7);arm=int(np.argmax(res[f]-allow[f]))
        oc=verified_oracle_contract()[0];oracle=FramewiseReachabilityOracle(g1,c,oc['oracle'],natural)
        seeds=[q[start:end].copy(),independent_anchors(oracle,t[start:end],h[start:end],natural)]
        for k,seed in enumerate(seeds):
            p=folder/f'FRAME_{f}_SEED_{k}.json';npz=folder/f'FRAME_{f}_SEED_{k}.npz'
            if p.exists():rows.append(read(p));continue
            print('TRUST_WINDOW_START',case,f,k,flush=True);clock=time.monotonic()
            value,info=solve_window(s,t[start:end],h[start:end],seed,allow[start:end],dt,arm,350)
            row=dict(frame=f,begin=start,end_exclusive=end,arm=arm,seed=k,info=info,runtime_s=time.monotonic()-clock)
            atomic_npz(npz,q=value,RAW_REPRESENTATION_TARGET=t[start:end],common_hand_q=h[start:end],source_timestamp=ts[start:end],allowance_m=allow[start:end])
            atomic_json(p,row);rows.append(row);print('TRUST_WINDOW_RESULT',case,f,k,info,flush=True)
    atomic_json(folder/'SUMMARY.json',dict(rows=rows,global_impossibility_proven=False,
        note='Relaxed necessary-condition diagnostic only; neither trust-region nor SLSQP finite failure is an infeasibility certificate.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
