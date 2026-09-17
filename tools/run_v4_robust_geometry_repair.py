#!/usr/bin/env python3
"""Bounded common collision repair with validated optimizer-only distances."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric,restore
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver

def run(case,family):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=V4/case/('robust_proxy_'+family+'_v1')
    src=V4/case/'reverse_geometry_continuation_v1/REVERSE_ANCHOR.npz' if family=='reverse' else RUN/case/'aggregate_step_semantics_diagnostic_v1/DETAILED_GEOMETRY_0.npz'
    q=np.load(src)['q'].copy();discovered={}
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(src),maximum_attempts=3,maximum_evaluations=400,
        classifier_unchanged=True,penalty_backend_only=file_record(ROOT/'tools/common_robust_proxy_penalty.py'),
        physical_limits_and_targets_unchanged=True,source_timestamps_unchanged=True))
    for attempt in range(4):
        old=folder/f'REPAIRED_{attempt-1}.npz'
        if attempt and old.exists():q=np.load(old)['q'].copy()
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        gp=folder/f'GEOMETRY_{attempt}.json'
        if gp.exists():geom=read(gp)['geometry']
        else:
            geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
            atomic_json(gp,dict(geometry=geom))
        blocked=[]
        for r in geom:
            bad=[x for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
            if bad:blocked.append(r['frame'])
            for x in bad:discovered.setdefault(r['frame'],set()).add(tuple(x['geom_pair']))
        atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=met,blocked_frames=blocked))
        print('ROBUST_CHECK',case,family,attempt,'cart',met['failed_frames'],'temporal',met['temporal']['pass_temporal'],'blocked',len(blocked),flush=True)
        if met['pass_numeric'] and not blocked:
            out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            passpath=V4/case/'SOURCE_POSITION_PASS.json'
            if not passpath.exists():atomic_json(passpath,dict(metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,
                acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'))
            atomic_json(folder/'SOURCE_POSITION_PASS.json',dict(trajectory=file_record(out),geometry=file_record(gp),metrics=met));return
        if attempt==3:break
        pp={};halo=(16,32,48)[attempt]
        for f,pairs in discovered.items():
            for k in range(max(0,f-halo),min(len(q),f+halo+1)):pp.setdefault(k,set()).update(pairs)
        pp={f:sorted(pairs) for f,pairs in pp.items()}
        out=folder/f'REPAIRED_{attempt}.npz';mp=folder/f'REPAIRED_{attempt}.json'
        if not out.exists():
            print('ROBUST_RESTORE',case,family,attempt,flush=True);start=time.monotonic()
            q,fit=restore(s,t,h,q,dt,allow,collision_pairs=pp,max_nfev=400)
            atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
            atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start,penalty_pairs={str(f):v for f,v in pp.items()}))
    atomic_json(folder/'COMPLETE.json',dict(metrics=met,blocked_frames=blocked,global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('family',choices=['reverse','saved']);a=p.parse_args();run(a.case,a.family)
