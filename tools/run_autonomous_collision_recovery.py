#!/usr/bin/env python3
"""Common detailed-contact discovery with temporally propagated posture repair."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_fixed_anchor_precision import restore as fixed_restore


def run(case,relative):
    verified_oracle_contract();g,c,natural=model();s=CommonPositionSolver(g,c,read(QUAL),natural);g.assign(natural);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)));folder=RUN/case/'collision_posture_precision_v1'
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];initial=RUN/case/relative;q=np.load(initial)['q'].copy()
    ap=RUN/case/'certified_anchor_recovery/CERTIFIED_ANCHORS.npz'
    if ap.exists():a=np.load(ap);anchors=a['q'].copy();mask=a['fixed_mask'].copy()
    else:anchors=q.copy();mask=np.zeros_like(q,dtype=bool)
    oldcfg=read(ROOT/'configs/common_g1_morphology_adapter_v1.json')
    halos=oldcfg['collision_temporal_projection']['window_padding_candidates_frames'][:3]
    atomic_json(folder/'CONTRACT.json',dict(source=file_record(initial),window_padding_frames=halos,
        padding_policy=file_record(ROOT/'configs/common_g1_morphology_adapter_v1.json'),
        collision_classifier_unchanged=True,geometry_tolerance_m=1e-5,maximum_attempts=3,
        maximum_evaluations_per_attempt=400,proxy_penalty_is_candidate_generation_only=True,
        final_acceptance='unchanged detailed geometry and raw/closest-feasible/temporal gates',
        implementation=file_record(ROOT/'tools/common_fixed_anchor_precision.py')))
    discovered={}
    for attempt,halo in enumerate(halos):
        before=folder/f'BEFORE_{attempt}_GEOMETRY.json'
        if before.exists():geometry=read(before)['geometry']
        else:
            geometry=[]
            for f,(v,hand) in enumerate(zip(q,h)):
                geometry.append(dict(frame=f,records=c.inspect(v,*hand)))
            atomic_json(before,dict(geometry=geometry))
        blocked=[]
        for row in geometry:
            bad=[r for r in row['records'] if r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
            if bad:blocked.append(row['frame'])
            for r in bad:discovered.setdefault(row['frame'],set()).add(tuple(r['geom_pair']))
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        if not blocked and met['pass_numeric']:
            atomic_npz(folder/'QUALIFIED_SOURCE_Q.npz',q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,
                EXECUTABLE_FK_POSITION=actual,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,
                SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert,common_hand_q=h,source_timestamp=ts)
            atomic_json(folder/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(folder/'QUALIFIED_SOURCE_Q.npz'),
                geometry=file_record(before),preparation_join='PENDING',common_execution_only=True))
            print('COLLISION_POSTURE_PASS',case,attempt,flush=True);return
        pairs={}
        for f,pp in discovered.items():
            for k in range(max(0,f-halo),min(len(q),f+halo+1)):pairs.setdefault(k,set()).update(pp)
        pairs={f:sorted(pp) for f,pp in pairs.items()}
        p=folder/f'ATTEMPT_{attempt}.npz';mp=folder/f'ATTEMPT_{attempt}.json'
        if p.exists():q=np.load(p)['q'].copy()
        else:
            print('COLLISION_POSTURE_START',case,attempt,'blocked',len(blocked),'halo',halo,flush=True);start=time.monotonic()
            q,fit=fixed_restore(s,t,h,q,dt,allow,collision_pairs=pairs,max_nfev=400,fixed_mask=mask,fixed_values=anchors)
            atomic_npz(p,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
            atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start,prior_blocked_frames=blocked,pairs={str(f):pp for f,pp in pairs.items()}))
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        row=read(mp);row['metrics']=met;atomic_json(mp,row)
        print('COLLISION_POSTURE_NUMERIC',case,attempt,met['pass_numeric'],len(met['failed_frames']),flush=True)
    geometry=[dict(frame=f,records=c.inspect(v,*hand)) for f,(v,hand) in enumerate(zip(q,h))]
    blocked=[row['frame'] for row in geometry if any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in row['records'])]
    atomic_json(folder/'FINAL_GEOMETRY.json',dict(geometry=geometry,blocked_frames=blocked,metrics=met))
    if not blocked and met['pass_numeric']:
        atomic_json(folder/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(p),geometry=file_record(folder/'FINAL_GEOMETRY.json'),preparation_join='PENDING'))
    print('COLLISION_POSTURE_COMPLETE',case,'blocked',len(blocked),'numeric',met['pass_numeric'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('relative');a=p.parse_args();run(a.case,a.relative)
