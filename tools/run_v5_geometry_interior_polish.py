#!/usr/bin/env python3
"""Common numerical interior polishing; physical gates are not changed."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_geometry_boundary_v5 import GeometryBoundarySolver
from tools.common_local_redundancy_v5 import solve

def run(case,source_path):
    verified_oracle_contract();g,c,n=model();s=GeometryBoundarySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=orbit_enclosures(g);t,h,ts,source=load_input(case,g,n)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];dt=float(np.median(np.diff(ts)))
    q=np.load(source_path)['q'].copy();folder=V5/case/'detailed_interior_polish_v1';tracked={}
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(Path(source_path)),common_iterations=300,common_attempts=3,
        position_search_interior_m=1e-7,geometry_search_interior='one unchanged geometry numerical tolerance mapped through local distance Jacobian',
        final_acceptance_unchanged=True,preparation_unchanged=True,targets_unchanged=True))
    prior=read(V5/case/'GEOMETRY_0.json')['geometry']
    for row in prior:
        for x in row['records']:
            if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY'):
                tracked.setdefault(row['frame'],set()).add(tuple(x['geom_pair']))
    for attempt in range(4):
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
        bad=[(r['frame'],x) for r in geom for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
        gp=folder/f'GEOMETRY_{attempt}.json';atomic_json(gp,dict(geometry=geom,metrics=met))
        print('INTERIOR_CHECK',case,attempt,'numeric',met['pass_numeric'],'failures',met['failed_frames'],'blocked',[f for f,x in bad],flush=True)
        if met['pass_numeric'] and not bad:
            out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            record=dict(case=case,metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,
                acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE')
            atomic_json(folder/'SOURCE_POSITION_PASS.json',record);atomic_json(V5/case/'SOURCE_POSITION_PASS.json',record);return
        if attempt==3:break
        for f,x in bad:tracked.setdefault(f,set()).add(tuple(x['geom_pair']))
        frame_solvers={};influenced=set()
        for f,pairs in tracked.items():
            local=GeometryBoundarySolver(g,c,read(QUAL),n)
            for pair in pairs:
                probe=q[f].copy();local.pose_jacobian(probe,h[f])
                from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver
                dist,jac=RobustProxyPenaltySolver.clearance_values(local,[pair]);control=int(np.argmax(np.abs(jac[0])));direction=np.sign(jac[0,control])
                # Start on the blocked side, whether the current q is clear or not.
                for displacement in (0,.005,.01,.02,.04,.08):
                    probe=q[f].copy();probe[control]-=direction*displacement
                    rr=c.inspect(probe,*h[f])
                    if any(tuple(x['geom_pair'])==pair and x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in rr):break
                record=local.discover(probe,h[f],pair)
                point=np.array(record['point']);point[control]+=direction*c.tolerance/max(abs(jac[0,control]),1e-6)
                record['point']=point.tolist();record['search_interior_rad']=float(c.tolerance/max(abs(jac[0,control]),1e-6))
                local.boundaries[pair]=record;influenced.update(record['influenced_joints'])
                atomic_json(folder/f'BOUNDARY_{attempt}_{f}_{pair[0]}_{pair[1]}.json',record)
            frame_solvers[f]=local
        class Dispatch(GeometryBoundarySolver):
            def clearance_values(self,pairs):
                local=frame_solvers[int(pairs[0][2])];local.current_q=self.current_q
                return local.clearance_values([p[:2] for p in pairs])
        ss=Dispatch(g,c,read(QUAL),n);begin=max(0,min(tracked)-24);end=min(len(q),max(tracked)+25)
        goals={f-begin:[(*p,f) for p in pairs] for f,pairs in tracked.items()};fits=[]
        for arm in sorted({k//7 for k in influenced}):
            search_allow=allow[begin:end].copy();search_allow[:,arm]-=1e-7
            # Endpoints are fixed; only active frames need the extra search inset.
            search_allow[:2]=allow[begin:begin+2];search_allow[-2:]=allow[end-2:end]
            start=time.monotonic();q[begin:end],fit=solve(ss,t[begin:end],h[begin:end],q[begin:end],search_allow,dt,arm,goals,300)
            fit['runtime_s']=time.monotonic()-start;fits.append(fit)
        atomic_npz(folder/f'CANDIDATE_{attempt}.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        atomic_json(folder/f'FIT_{attempt}.json',dict(fits=fits,window=[begin,end]))
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='LOCAL_REDUNDANCY_COLLISION_INFEASIBILITY',global_impossibility_proven=False,next='COMMON_CLOSEST_FEASIBLE_REALIZATION',metrics=met))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
