#!/usr/bin/env python3
"""Bounded common frame-independent search; no failed search is a certificate.

Uses unchanged targets, joint limits and final geometry classification. Labels
only locate input/output artifacts. The search does not accept a trajectory.
"""
from pathlib import Path
import sys,time
import numpy as np
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def run(case,path):
    oc=verified_oracle_contract()[0];g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],n);g.assign(n);bounds=full_chain_enclosures(g)
    t,h,ts,source=load_input(case,g,n);path=Path(path).resolve();q=np.load(path)['q']
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    m,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
    failed=np.flatnonzero(np.any(res>allow,axis=1));clusters=[]
    for f in failed:
        if not clusters or f>clusters[-1][-1]+1:clusters.append([int(f)])
        else:clusters[-1].append(int(f))
    selected=set()
    for rows in clusters:
        selected.update([rows[0],rows[len(rows)//2],rows[-1],max(rows,key=lambda f:float(np.max(res[f]-allow[f])))])
    folder=path.parent/(path.stem+'_global_position_audit_v1')
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(path),source=source,selected_frames=sorted(selected),
        common_halton_seeds=64,maximum_evaluations_per_seed=300,geometry_candidate_budget=12,
        raw_targets_changed=False,acceptance_changed=False,global_infeasibility_proven=False))
    seeds=oracle.lower+(oracle.upper-oracle.lower)*qmc.Halton(14,scramble=False).random(64)
    results=[]
    for f in sorted(selected):
        dest=folder/f'FRAME_{f:04d}.json'
        if dest.exists():results.append(read(dest));continue
        start=time.monotonic();candidates=[]
        for sid,seed in enumerate([q[f],n,*seeds]):
            v,fit=oracle.fit(t[f],h[f],seed,max_evaluations=300)
            error=np.array(fit['residual_m']);key=float(np.maximum(error-allow[f],0).max())
            candidates.append(dict(seed_id=sid,q=v.tolist(),residual_m=error.tolist(),excess_m=key,fit=fit))
        candidates.sort(key=lambda r:(r['excess_m'],sum(r['residual_m']),r['seed_id']))
        tested=[];witness=None
        for row in candidates[:12]:
            records=c.inspect(np.array(row['q']),*h[f])
            valid=not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in records)
            tested.append(dict(**row,geometry=records,geometry_valid=valid))
            if valid and row['excess_m']==0:witness=tested[-1];break
        result=dict(frame=f,candidates=candidates,geometry_tested=tested,witness=witness,
            allowance_m=allow[f].tolist(),static_lower_bound_m=lb[f].tolist(),runtime_s=time.monotonic()-start,
            status='FRAMEWISE_EXECUTABLE_WITNESS' if witness else 'BOUNDED_SEARCH_NO_VALID_WITNESS',global_infeasibility_proven=False)
        atomic_json(dest,result);results.append(result)
        print('COMMON_GLOBAL_POSITION_AUDIT',case,f,result['status'],'best_mm',np.array(candidates[0]['residual_m'])*1000,flush=True)
    atomic_json(folder/'COMPLETE.json',dict(case=case,frames=[r['frame'] for r in results],
        witnessed=[r['frame'] for r in results if r['witness']],no_witness=[r['frame'] for r in results if not r['witness']],
        global_infeasibility_proven=False,source=source,input=file_record(path)))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
