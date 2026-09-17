#!/usr/bin/env python3
"""Common redundant-DOF repair holding collision-controlling coordinates fixed."""
from pathlib import Path
import sys,time,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver
from tools.common_near_boundary_precision_v5 import restore

def run(case):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=orbit_enclosures(g);t,h,ts,source=load_input(case,g,n)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];dt=float(np.median(np.diff(ts)))
    src=V5/case/'near_boundary_precision_v1/CANDIDATE_2.npz';q=np.load(src)['q'].copy()
    folder=V5/case/'collision_invariant_recovery_v1';controlled=set();blocked=[]
    for row in read(V5/case/'GEOMETRY_0.json')['geometry']:
        for x in row['records']:
            if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY'):
                f=row['frame'];s.pose_jacobian(q[f],h[f]);_,jac=s.clearance_values([tuple(x['geom_pair'])]);controlled.update(np.flatnonzero(np.abs(jac[0])>1e-9).tolist());blocked.append(f)
    begin=max(0,min(blocked)-48);end=min(len(q),max(blocked)+49);mask=np.ones_like(q,bool)
    for arm in sorted({k//7 for k in controlled}):mask[begin+2:end-2,arm*7:arm*7+6]=False
    mask[:,list(controlled)]=True;fixed=q.copy();ss=copy.copy(s);ss.config=dict(s.config)
    ss.config['maximum_step_norm_rad']=np.sqrt(14)*min(ss.config['maximum_joint_step_rad'],ss.config['maximum_velocity_rad_s']*dt)
    atomic_json(folder/'CONTRACT.json',dict(source=file_record(src),controlled_joint_indices=sorted(controlled),
        rule='Current collision-free coordinate values held fixed; other redundant arm coordinates optimize unchanged raw targets under common constraints',
        methods_or_link_whitelists=False,window=[begin,end],iterations=3,maximum_function_evaluations=400))
    for attempt in range(4):
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))]
        bad=[r['frame'] for r in geom if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        gp=folder/f'GEOMETRY_{attempt}.json';atomic_json(gp,dict(geometry=geom,metrics=met))
        print('COLLISION_INVARIANT_REPAIR',case,attempt,met['pass_numeric'],met['failed_frames'],'blocked',bad,flush=True)
        if met['pass_numeric'] and not bad:
            out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            record=dict(case=case,metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,
                acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE')
            atomic_json(folder/'SOURCE_POSITION_PASS.json',record)
            if not (V5/case/'SOURCE_POSITION_PASS.json').exists():atomic_json(V5/case/'SOURCE_POSITION_PASS.json',record)
            return
        if attempt==3:break
        start=time.monotonic();q,fit=restore(ss,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=fixed)
        np.testing.assert_array_equal(q[:,list(controlled)],fixed[:,list(controlled)])
        atomic_npz(folder/f'CANDIDATE_{attempt}.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        atomic_json(folder/f'FIT_{attempt}.json',dict(fit=fit,runtime_s=time.monotonic()-start))
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='CLOSEST_FEASIBLE_SOLVER_NOT_CONVERGED',metrics=met,blocked_frames=bad))

if __name__=='__main__':run(sys.argv[1])
