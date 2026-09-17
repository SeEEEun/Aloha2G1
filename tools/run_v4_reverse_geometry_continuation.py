#!/usr/bin/env python3
"""Common future-feasible-anchor branch continuation; no target edits."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric,restore

def run(case):
    oc=verified_oracle_contract()[0];g,c,natural=model();s=CommonPositionSolver(g,c,read(QUAL),natural);g.assign(natural);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],natural)
    folder=V4/case/'reverse_geometry_continuation_v1';initial=V4/case/'REPAIRED_2.npz';old=np.load(initial)['q'].copy()
    met,actual,res,lb,cert,allow=qualify_numeric(s,old,t,h,ts,bounds,slack)
    seedpath=folder/'REVERSE_ANCHOR.npz'
    if seedpath.exists():q=np.load(seedpath)['q'].copy()
    else:
        bad=read(V4/case/'CHECK_0.json')['blocked_frames'];anchor=next(f for f in range(max(bad)+1,len(old)) if max(res[f]-allow[f])<=0)
        q=old.copy();previous=q[anchor].copy();events=[]
        for f in range(anchor-1,-1,-1):
            candidates=[]
            for j,seed in enumerate((previous,old[f],natural)):
                v,fit=oracle.fit(t[f],h[f],seed,max_evaluations=160,reference=previous,posture_weight=.001)
                rr=c._records(v,*h[f]);error=np.linalg.norm(s.pose_jacobian(v,h[f])[0]-t[f],axis=1)
                if rr:
                    pairs=sorted({tuple(r['geom_pair']) for r in rr})
                    v,fix=s.optimize(t[f],h[f],v,s.lower,s.upper,previous,previous,v,pairs,100)
                    rr=c._records(v,*h[f]);error=np.linalg.norm(s.pose_jacobian(v,h[f])[0]-t[f],axis=1)
                key=(bool(np.any(error>allow[f])),bool(rr),float(np.linalg.norm(v-previous)),float(error.max()))
                candidates.append((key,v,j,error,rr))
                if not key[0] and not key[1] and j==0:break
            candidates.sort(key=lambda v:v[0]);key,v,j,error,rr=candidates[0];q[f]=v;previous=v
            events.append(dict(frame=f,seed=j,error_mm=(error*1000).tolist(),blocked=rr))
            if f%30==0:print('REVERSE_GEOMETRY',case,f,'residual',error.max()*1000,'blocked',len(rr),flush=True)
        q[:,[6,13]]=old[0,[6,13]]
        atomic_npz(seedpath,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        atomic_json(folder/'REVERSE_ANCHOR.json',dict(anchor=anchor,events=events,source=file_record(initial),implementation=file_record(Path(__file__))))
    discovered={}
    for attempt in range(4):
        p=folder/f'REPAIRED_{attempt-1}.npz'
        if attempt and p.exists():q=np.load(p)['q'].copy()
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
        print('REVERSE_CHECK',case,attempt,met['pass_numeric'],'cart',met['failed_frames'],'blocked',len(blocked),flush=True)
        if met['pass_numeric'] and not blocked:
            out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            atomic_json(V4/case/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,
                acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'));return
        if attempt==3:break
        pairs={};halo=(16,32,48)[attempt]
        for f,pp in discovered.items():
            for k in range(max(0,f-halo),min(len(q),f+halo+1)):pairs.setdefault(k,set()).update(pp)
        pairs={f:sorted(pp) for f,pp in pairs.items()}
        out=folder/f'REPAIRED_{attempt}.npz';mp=folder/f'REPAIRED_{attempt}.json'
        if not out.exists():
            print('REVERSE_RESTORE',case,attempt,flush=True);start=time.monotonic()
            q,fit=restore(s,t,h,q,dt,allow,collision_pairs=pairs,max_nfev=400)
            atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
            atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start))
    atomic_json(folder/'COMPLETE.json',dict(metrics=met,blocked_frames=blocked,global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
