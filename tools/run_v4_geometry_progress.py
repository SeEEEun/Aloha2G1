#!/usr/bin/env python3
"""Method-blind detailed-geometry-driven minimum-change continuation."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric,restore
from tools.common_geometry_progress_penalty import GeometryProgressPenaltySolver

def run(case):
    verified_oracle_contract();g,c,n=model();s=GeometryProgressPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=V4/case/'detailed_geometry_progress_v1';src=RUN/case/'aggregate_step_semantics_diagnostic_v1/DETAILED_GEOMETRY_0.npz';q=np.load(src)['q'].copy();goals={}
    ap=RUN/case/'certified_anchor_recovery/CERTIFIED_ANCHORS.npz';z=np.load(ap);mask=z['fixed_mask'];anchors=z['q']
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(src),maximum_attempts=6,maximum_evaluations=400,
        rule='Local optimizer proxy-distance increment equals current detailed penetration lower bound, at least common numerical geometry tolerance. Unresolved geometry instead receives half current proxy penetration plus numerical tolerance. Goals are NOT acceptance thresholds. Every resulting frame receives unchanged detailed classification.',
        local_goal_scope='optimization-only, frame/pair geometry state; no method, representation, task outcome input',
        certified_anchor_seed_policy='first three attempts hold proven near-optimal certified q; next three release q while preserving exact same position allowance',
        implementation=file_record(ROOT/'tools/common_geometry_progress_penalty.py')))
    for attempt in range(7):
        prev=folder/f'REPAIRED_{attempt-1}.npz'
        if attempt and prev.exists():q=np.load(prev)['q'].copy()
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
            for x in bad:
                f=r['frame'];pair=tuple(x['geom_pair']);s.pose_jacobian(q[f],h[f]);distance=s.clearance_values([pair])[0][0]
                depth=(x.get('detailed_penetration_lower_bound_mm') or 0)/1000
                step=max(depth,c.tolerance) if x['classification']=='HARD_SELF_COLLISION' else max(-distance/2,0)+c.tolerance
                goals.setdefault(f,{})[pair]=float(distance+step)
        atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=met,blocked_frames=blocked,local_search_goals={str(f):[list(p)+[v] for p,v in pp.items()] for f,pp in goals.items()}))
        print('DETAILED_PROGRESS_CHECK',case,attempt,'numeric',met['pass_numeric'],'cart',met['failed_frames'],'blocked',len(blocked),flush=True)
        if met['pass_numeric'] and not blocked:
            out=folder/'QUALIFIED_SOURCE_Q.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            passpath=V4/case/'SOURCE_POSITION_PASS.json'
            if not passpath.exists():atomic_json(passpath,dict(metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'))
            atomic_json(folder/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(out),geometry=file_record(gp)));return
        if attempt==6:break
        pp={f:[(*p,v) for p,v in pairs.items()] for f,pairs in goals.items()}
        out=folder/f'REPAIRED_{attempt}.npz';mp=folder/f'REPAIRED_{attempt}.json'
        if not out.exists():
            print('DETAILED_PROGRESS_RESTORE',case,attempt,flush=True);start=time.monotonic()
            q,fit=restore(s,t,h,q,dt,allow,collision_pairs=pp,max_nfev=400,fixed_mask=mask if attempt<3 else None,fixed_values=anchors if attempt<3 else None)
            atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
            atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start))
    atomic_json(folder/'COMPLETE.json',dict(metrics=met,blocked_frames=blocked,global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
