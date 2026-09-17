#!/usr/bin/env python3
"""Sparse feasibility restoration using saved detailed-geometry tangents."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_geometry_boundary_v5 import GeometryBoundarySolver
from tools.common_physical_position_v4 import restore

def run(case):
    verified_oracle_contract();g,c,n=model();s=GeometryBoundarySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=orbit_enclosures(g);t,h,ts,source=load_input(case,g,n)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];dt=float(np.median(np.diff(ts)))
    src=V5/case/'detailed_boundary_repair_v1/CANDIDATE_0.npz';q=np.load(src)['q'].copy()
    folder=V5/case/'boundary_sparse_precision_v1';frame_solvers={};influenced=set();goals={}
    for p in sorted((V5/case/'detailed_interior_polish_v1').glob('BOUNDARY_0_*.json')):
        _,_,frame,ga,gb=p.stem.split('_');f=int(frame);pair=(int(ga),int(gb));record=read(p)
        local=frame_solvers.setdefault(f,GeometryBoundarySolver(g,c,read(QUAL),n));local.boundaries[pair]=record
        influenced.update(record['influenced_joints']);goals.setdefault(f,[]).append((*pair,f))
    class Dispatch(GeometryBoundarySolver):
        def clearance_values(self,pairs):
            local=frame_solvers[int(pairs[0][2])];local.current_q=self.current_q
            return local.clearance_values([p[:2] for p in pairs])
    ss=Dispatch(g,c,read(QUAL),n)
    begin=max(0,min(goals)-48);end=min(len(q),max(goals)+49)
    mask=np.ones_like(q,bool)
    for arm in sorted({k//7 for k in influenced}):mask[begin+2:end-2,arm*7:arm*7+6]=False
    for attempt in range(4):
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
        bad=[(r['frame'],x) for r in geom for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
        gp=folder/f'GEOMETRY_{attempt}.json';atomic_json(gp,dict(geometry=geom,metrics=met))
        print('BOUNDARY_PRECISION',case,attempt,'numeric',met['pass_numeric'],'failed',met['failed_frames'],'blocked',[f for f,x in bad],flush=True)
        if met['pass_numeric'] and not bad:
            out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            record=dict(case=case,metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,
                acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE')
            atomic_json(folder/'SOURCE_POSITION_PASS.json',record)
            if not (V5/case/'SOURCE_POSITION_PASS.json').exists():atomic_json(V5/case/'SOURCE_POSITION_PASS.json',record)
            return
        if attempt==3:break
        start=time.monotonic();q,fit=restore(ss,t,h,q,dt,allow,collision_pairs=goals,max_nfev=400,fixed_mask=mask,fixed_values=q)
        atomic_npz(folder/f'CANDIDATE_{attempt}.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        atomic_json(folder/f'FIT_{attempt}.json',dict(fit=fit,runtime_s=time.monotonic()-start,window=[begin,end]))
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='LOCAL_REDUNDANCY_COLLISION_INFEASIBILITY',global_impossibility_proven=False,next='COMMON_CLOSEST_FEASIBLE_REALIZATION',metrics=met))

if __name__=='__main__':run(sys.argv[1])
