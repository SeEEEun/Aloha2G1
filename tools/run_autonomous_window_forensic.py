#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_hard_window_search import solve_window


def run(case):
    cfg=read(QUAL);g1,c,natural=model();s=CommonPositionSolver(g1,c,cfg,natural);g1.assign(natural);b=orbit_enclosures(g1)
    t,h,ts,source=load_input(case,g1,natural);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    sourceq=RUN/case/'dual_position_feasible_history_v1/RESTORED_2.npz';q=np.load(sourceq)['q'].copy()
    metrics,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,b,slack)
    folder=RUN/case/'hard_window_forensic';rows=[]
    # Select first/worst remaining positional failure, before new window results.
    failed=metrics['failed_frames'];maxframe=int(np.argmax(np.max(res-allow,axis=1)))
    frames=list(dict.fromkeys([failed[0],maxframe,failed[-1]])) if failed else []
    selection=dict(source=file_record(sourceq),frames=frames,padding_before=12,padding_after=6,seed_count=3,
        max_iterations_per_seed=400,arm_selection='wrist with maximum allowance excess',
        relaxation='independent arm; free window boundaries; no collision restrictions; step norm relaxed from bilateral to arm-only')
    atomic_json(folder/'WINDOW_SEARCH_CONTRACT.json',selection)
    for f in frames:
        start=max(0,f-12);end=min(len(q),f+7);arm=int(np.argmax(res[f]-allow[f]));base=q[start:end].copy()
        # Previous restored state, independent natural framewise anchors, and a
        # constant closest-frame anchor. All are numerical seeds, not commands.
        oc=verified_oracle_contract()[0];oracle=FramewiseReachabilityOracle(g1,c,oc['oracle'],natural)
        seeds=[base,independent_anchors(oracle,t[start:end],h[start:end],natural),np.repeat(base[f-start][None],len(base),axis=0)]
        for k,seed in enumerate(seeds):
            mp=folder/f'FRAME_{f}_SEED_{k}.json';qp=folder/f'FRAME_{f}_SEED_{k}.npz'
            if mp.exists():row=read(mp);rows.append(row);continue
            print('HARD_WINDOW_START',case,f,k,flush=True);clock=time.monotonic()
            fit,info=solve_window(s,t[start:end],h[start:end],seed,allow[start:end],dt,arm,400)
            row=dict(frame=f,begin=start,end_exclusive=end,arm=arm,seed=k,info=info,runtime_s=time.monotonic()-clock)
            atomic_npz(qp,q=fit,RAW_REPRESENTATION_TARGET=t[start:end],common_hand_q=h[start:end],source_timestamp=ts[start:end],allowance_m=allow[start:end])
            atomic_json(mp,row);rows.append(row);print('HARD_WINDOW_RESULT',case,f,k,info,flush=True)
    atomic_json(folder/'SUMMARY.json',dict(rows=rows,global_impossibility_proven=False,
        interpretation='Finite hard-constraint search; any witness still needs bilateral continuity, boundary and collision checks.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
