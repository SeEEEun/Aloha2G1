#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric
from tools.common_feasible_geometry_window import solve
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver

def run(case):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=V4/case/'keep_feasible_geometry_window_v1';src=RUN/case/'aggregate_step_semantics_diagnostic_v1/DETAILED_GEOMETRY_0.npz';q=np.load(src)['q'].copy()
    met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack);assert met['pass_numeric']
    contract=read(V4/case/'hard_physical_geometry_window_v1/CONTRACT.json');begin,end=contract['window'];pairs=[tuple(p) for p in contract['pairs']]
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(src),physical_limits_unchanged=True,maximum_iterations=500,window=[begin,end],arms=contract['arms'],
        keep_feasible='hard Cartesian/closest-feasible, per-joint velocity/acceleration and limits; no aggregate cap',implementation=file_record(ROOT/'tools/common_feasible_geometry_window.py')))
    out=folder/'CANDIDATE.npz'
    if out.exists():q=np.load(out)['q'];fits=read(folder/'FIT.json')['fits']
    else:
        fits=[]
        for arm in contract['arms']:
            start=time.monotonic();q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,pairs,500)
            fit['runtime_s']=time.monotonic()-start;fits.append(fit)
        atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts);atomic_json(folder/'FIT.json',dict(fits=fits))
    met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
    geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
    blocked=[r['frame'] for r in geom if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
    qualified=bool(met['pass_numeric'] and not blocked);mp=folder/'RESULT.json'
    atomic_json(mp,dict(metrics=met,geometry=geom,blocked_frames=blocked,qualified=qualified,trajectory=file_record(out),fits=fits,global_impossibility_proven=False))
    print('KEEP_FEASIBLE_RESULT',case,'numeric',met['pass_numeric'],'blocked',len(blocked),flush=True)
    if qualified:atomic_json(V4/case/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(out),geometry=file_record(mp),source=source,acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
