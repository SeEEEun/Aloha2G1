#!/usr/bin/env python3
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_geometry_boundary_v5 import GeometryBoundarySolver
from tools.common_local_redundancy_v5 import solve

def run(case):
    verified_oracle_contract();g,c,n=model();s=GeometryBoundarySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=orbit_enclosures(g);t,h,ts,source=load_input(case,g,n)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];dt=float(np.median(np.diff(ts)))
    source_record=read(V5/case/'INPUT.json')['trajectory'];initial=np.load(source_record['path'])['q'].copy();q=initial.copy()
    folder=V5/case/'detailed_boundary_repair_v1'
    for attempt in range(3):
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
        bad=[(r['frame'],x) for r in geom for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
        gp=folder/f'GEOMETRY_{attempt}.json';atomic_json(gp,dict(geometry=geom,metrics=met))
        print('DETAILED_BOUNDARY_CHECK',case,attempt,'numeric',met['pass_numeric'],'blocked',[f for f,x in bad],flush=True)
        if met['pass_numeric'] and not bad:
            out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            atomic_json(folder/'SOURCE_POSITION_PASS.json',dict(case=case,metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source));return
        if attempt==2:break
        if not met['pass_numeric']:q=initial.copy()
        # A separate solver holds a geometry tangent for each constrained frame.
        frame_solvers={};influenced=set()
        for f,x in bad:
            local=GeometryBoundarySolver(g,c,read(QUAL),n)
            record=local.discover(q[f],h[f],tuple(x['geom_pair']))
            frame_solvers[f]=local;influenced.update(record['influenced_joints'])
            atomic_json(folder/f'BOUNDARY_{attempt}_{f}.json',record)
            print('DETAILED_BOUNDARY',f,'control',record['control_joint'],'delta',np.array(record['point'])-q[f],flush=True)
        class Dispatch(GeometryBoundarySolver):
            def pose_jacobian(self,value,hh):
                self.current_q=np.array(value).copy();return super().pose_jacobian(value,hh)
            def clearance_values(self,pairs):
                # The third index references the local geometric model, not a method.
                local=frame_solvers[int(pairs[0][2])];local.current_q=self.current_q
                return local.clearance_values([p[:2] for p in pairs])
        ss=Dispatch(g,c,read(QUAL),n);begin=max(0,min(frame_solvers)-48);end=min(len(q),max(frame_solvers)+49)
        goals={f-begin:[(*tuple(x['geom_pair']),f)] for f,x in bad}
        fits=[]
        for arm in sorted({k//7 for k in influenced}):
            start=time.monotonic();q[begin:end],fit=solve(ss,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,goals,300)
            fit['runtime_s']=time.monotonic()-start;fits.append(fit)
        atomic_npz(folder/f'CANDIDATE_{attempt}.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        atomic_json(folder/f'FIT_{attempt}.json',dict(fits=fits,window=[begin,end]))
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='LOCAL_REDUNDANCY_COLLISION_INFEASIBILITY',global_impossibility_proven=False,next='COMMON_CLOSEST_FEASIBLE_REALIZATION',metrics=met))

if __name__=='__main__':run(sys.argv[1])
