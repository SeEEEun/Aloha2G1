#!/usr/bin/env python3
"""Independent hard-constrained retry of collision/temporal transition windows."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric
from tools.common_physical_hard_window import solve
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver

def run(case):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=V4/case/'hard_physical_geometry_window_v1';src=RUN/case/'aggregate_step_semantics_diagnostic_v1/DETAILED_GEOMETRY_0.npz';base=np.load(src)['q'].copy()
    met,act,res,lb,cert,allow=qualify_numeric(s,base,t,h,ts,bounds,slack)
    geo=read(V4/case/'GEOMETRY_0.json')['geometry'];bad=[r for r in geo if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
    pairs=sorted({tuple(x['geom_pair']) for r in bad for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')})
    seeds=[src,V4/case/'reverse_geometry_continuation_v1/REVERSE_ANCHOR.npz',V4/case/'robust_proxy_reverse_v1/REPAIRED_2.npz']
    # Determine movable arm from model Jacobian influence of actual bad pairs,
    # not a case/representation label. Cross-arm cases retain both candidates.
    influence=np.zeros(14)
    for r in bad:
        s.pose_jacobian(base[r['frame']],h[r['frame']]);influence+=np.abs(s.clearance_values(pairs)[1]).sum(axis=0)
    arms=[a for a in range(2) if influence[a*7:(a+1)*7].sum()>1e-12]
    begin=max(0,min(r['frame'] for r in bad)-32);end=min(len(base),max(r['frame'] for r in bad)+33)
    atomic_json(folder/'CONTRACT.json',dict(window=[begin,end],arms=arms,pairs=pairs,maximum_seeds=3,maximum_iterations=500,
        fixed_boundary_frames=2,source=file_record(src),aggregate_step_constraint=False,physical_limits_unchanged=True,
        objective='common proxy collision penalty; hard constraints are unchanged Cartesian/closest-feasible positions, joint limits and per-joint temporal limits',implementation=file_record(ROOT/'tools/common_physical_hard_window.py')))
    for k,seed in enumerate(seeds):
        out=folder/f'CANDIDATE_{k}.npz';mp=folder/f'CANDIDATE_{k}.json'
        if mp.exists():
            if read(mp)['qualified']:return
            continue
        q=base.copy();q[begin:end]=np.load(seed)['q'][begin:end];q[begin:begin+2]=base[begin:begin+2];q[end-2:end]=base[end-2:end];fits=[]
        for arm in arms:
            print('HARD_PHYSICAL_WINDOW_START',case,k,arm,begin,end,flush=True);start=time.monotonic()
            q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,pairs,500)
            fit['runtime_s']=time.monotonic()-start;fits.append(fit)
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
        blocked=[r['frame'] for r in geom if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        qualified=bool(met['pass_numeric'] and not blocked)
        atomic_json(mp,dict(metrics=met,geometry=geom,blocked_frames=blocked,qualified=qualified,fits=fits,trajectory=file_record(out)))
        print('HARD_PHYSICAL_WINDOW_RESULT',case,k,'cart',met['failed_frames'],'temporal',met['temporal']['pass_temporal'],'blocked',len(blocked),flush=True)
        if qualified:
            atomic_json(V4/case/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(out),geometry=file_record(mp),source=source,acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'));return
    atomic_json(folder/'COMPLETE.json',dict(status='BOUNDED_COMMON_HARD_CONSTRAINT_SEARCH_COMPLETED',global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
