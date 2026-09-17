#!/usr/bin/env python3
"""Common local detailed-geometry progress from improved feasible seeds."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric
from tools.common_feasible_local_geometry_window import solve
from tools.common_geometry_progress_penalty import GeometryProgressPenaltySolver

def run(case):
    verified_oracle_contract();g,c,n=model();s=GeometryProgressPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=V4/case/'local_feasible_detailed_geometry_v1'
    for seed in range(3):
        sf=V4/case/'untouched_dof_splice_fix_v1'/f'SEED_{seed}';src=sf/'POLISHED_0.npz';q=np.load(src)['q'].copy()
        q[:,[6,13]]=read(sf/'RANKING_1.json')['rows'][0]['wrist_joint_values'];sub=folder/f'SEED_{seed}';goals={}
        for attempt in range(4):
            old=sub/f'CANDIDATE_{attempt-1}.npz'
            if attempt and old.exists():q=np.load(old)['q'].copy()
            met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            assert met['pass_numeric'],'Keep-feasible candidate lost a physical/Cartesian constraint; do not promote'
            gp=sub/f'GEOMETRY_{attempt}.json'
            if gp.exists():geom=read(gp)['geometry']
            else:
                geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(q,h))];atomic_json(gp,dict(geometry=geom))
            blocked=[];influence=np.zeros(14)
            for row in geom:
                bad=[x for x in row['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
                if bad:blocked.append(row['frame'])
                for x in bad:
                    f=row['frame'];pair=tuple(x['geom_pair']);s.pose_jacobian(q[f],h[f]);distance,jac=s.clearance_values([pair]);influence+=np.abs(jac).sum(axis=0)
                    depth=(x.get('detailed_penetration_lower_bound_mm') or 0)/1000
                    step=max(depth,c.tolerance) if x['classification']=='HARD_SELF_COLLISION' else max(-distance[0]/2,0)+c.tolerance
                    goals.setdefault(f,{})[pair]=float(distance[0]+step)
            atomic_json(sub/f'CHECK_{attempt}.json',dict(metrics=met,blocked_frames=blocked))
            print('LOCAL_FEASIBLE_GEOMETRY_CHECK',case,seed,attempt,'blocked',blocked,flush=True)
            if not blocked:
                out=sub/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                    EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
                atomic_json(V4/case/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'));return
            if attempt==3:break
            begin=max(0,min(goals)-32);end=min(len(q),max(goals)+33);arms=[a for a in range(2) if influence[a*7:(a+1)*7].sum()>1e-12]
            pairs={f-begin:[(*p,v) for p,v in pp.items()] for f,pp in goals.items()};out=sub/f'CANDIDATE_{attempt}.npz';mp=sub/f'FIT_{attempt}.json'
            if not out.exists():
                fits=[]
                for arm in arms:
                    start=time.monotonic();q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,pairs,300)
                    fit['runtime_s']=time.monotonic()-start;fits.append(fit)
                atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
                atomic_json(mp,dict(fits=fits,window=[begin,end],arms=arms,local_goals={str(f):p for f,p in pairs.items()},implementation=file_record(ROOT/'tools/common_feasible_local_geometry_window.py')))
    atomic_json(folder/'COMPLETE.json',dict(status='BOUNDED_FEASIBLE_LOCAL_GEOMETRY_SEARCH_COMPLETED',global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
