#!/usr/bin/env python3
"""Common local exact-linear-algebra retry with locked external boundaries."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_fixed_anchor_dense import restore as fixed_restore


def invalid_frames(q,metrics,dt,cfg):
    delta=np.diff(q,axis=0);dd=np.diff(q,n=2,axis=0)
    frames=set(metrics['failed_frames'])
    for f in np.flatnonzero((np.max(np.abs(delta),axis=1)>min(cfg['maximum_joint_step_rad'],cfg['maximum_velocity_rad_s']*dt))|(np.linalg.norm(delta,axis=1)>cfg['maximum_step_norm_rad'])):frames.update([int(f),int(f+1)])
    for f in np.flatnonzero(np.max(np.abs(dd),axis=1)>cfg['maximum_acceleration_rad_s2']*dt*dt):frames.update([int(f),int(f+1),int(f+2)])
    return sorted(frames)


def windows(frames,halo,n):
    rows=[]
    for f in frames:
        a=max(0,f-halo);b=min(n,f+halo+1)
        if rows and a<=rows[-1][1]:rows[-1][1]=max(rows[-1][1],b)
        else:rows.append([a,b])
    return rows


def run(case):
    verified_oracle_contract();g,c,natural=model();cfg=read(QUAL);s=CommonPositionSolver(g,c,cfg,natural);g.assign(natural);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)));folder=RUN/case/'local_dense_continuation_v1'
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    initial=RUN/case/'certified_anchor_precision_v2/ATTEMPT_2.npz';q=np.load(initial)['q'].copy()
    a=np.load(RUN/case/'certified_anchor_recovery/CERTIFIED_ANCHORS.npz');anchors=a['q'].copy();fixed=a['fixed_mask'].copy()
    halos=read(ROOT/'configs/common_g1_morphology_adapter_v1.json')['collision_temporal_projection']['window_padding_candidates_frames'][:3]
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(initial),window_padding_frames=halos,max_evaluations_per_window=300,
        exact_dense_trust_region_linear_solver=True,boundary_rule='lock first/last two window states to preserve external velocity and acceleration',
        scope='common deterministic numerical retry; no acceptance or raw-target changes',implementation=file_record(ROOT/'tools/common_fixed_anchor_dense.py')))
    for attempt,halo in enumerate(halos):
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        ranges=windows(invalid_frames(q,met,dt,cfg),halo,len(q))
        if met['pass_numeric']:break
        atomic_json(folder/f'WINDOWS_{attempt}.json',dict(ranges=ranges,metrics=met))
        for k,(start,end) in enumerate(ranges):
            p=folder/f'PASS_{attempt}_WINDOW_{k}.npz';mp=folder/f'PASS_{attempt}_WINDOW_{k}.json'
            if p.exists():q[start:end]=np.load(p)['q'];continue
            seed=q[start:end].copy();mask=fixed[start:end].copy();fixedq=anchors[start:end].copy()
            if start>0:mask[:2]=True;fixedq[:2]=seed[:2]
            if end<len(q):mask[-2:]=True;fixedq[-2:]=seed[-2:]
            print('LOCAL_DENSE_START',case,attempt,start,end,flush=True);clock=time.monotonic()
            fit,info=fixed_restore(s,t[start:end],h[start:end],seed,dt,allow[start:end],max_nfev=300,fixed_mask=mask,fixed_values=fixedq)
            q[start:end]=fit
            atomic_npz(p,q=fit,RAW_REPRESENTATION_TARGET=t[start:end],common_hand_q=h[start:end],source_timestamp=ts[start:end])
            atomic_json(mp,dict(fit=info,runtime_s=time.monotonic()-clock,begin=start,end_exclusive=end))
            print('LOCAL_DENSE_WINDOW',case,attempt,k,info,flush=True)
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        p=folder/f'PASS_{attempt}_FULL.npz';atomic_npz(p,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        atomic_json(folder/f'PASS_{attempt}_FULL.json',dict(metrics=met,trajectory=file_record(p)))
        print('LOCAL_DENSE_RESULT',case,attempt,len(met['failed_frames']),met['temporal'],flush=True)
        if met['pass_numeric']:
            atomic_json(folder/'NUMERIC_PASS.json',dict(trajectory=file_record(p),metrics=met,next='common collision qualification'));return
    atomic_json(folder/'BOUNDED_SEARCH_COMPLETE.json',dict(metrics=met,global_impossibility_proven=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
