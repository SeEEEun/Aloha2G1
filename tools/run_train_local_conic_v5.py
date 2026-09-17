#!/usr/bin/env python3
"""Bounded common local hard-temporal search; independent final qualification."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_conic_trust_realization_v5 import solve

def run(case):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=full_chain_enclosures(g);t,h,ts,source=load_input(case,g,n)
    dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=TRAIN/case/'common_local_conic_v1';candidates=[]
    for p in (TRAIN/case).rglob('CHECK_*.json'):
        r=read(p)
        if 'trajectory' not in r:continue
        qp=Path(r['trajectory']['path'])
        if not qp.exists():continue
        m=r['metrics'];tm=m['temporal']
        score=(max(tm['maximum_velocity_rad_s']/4.5,tm['maximum_acceleration_rad_s2']/130)-1,len(m['failed_frames']))
        candidates.append((max(score[0],0)+score[1]/1000,str(qp)))
    candidates.sort();src=Path(candidates[0][1]);q=np.load(src)['q'].copy()
    atomic_json(folder/'INPUT.json',dict(source=source,trajectory=file_record(src),selection='minimum physical temporal excess plus failed-frame fraction; no method/outcome input',
        common_window_padding=24,common_iterations=300,common_trust_radius=.02,acceptance_unchanged=True))
    for attempt in range(3):
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=met))
        print('TRAIN_LOCAL_CONIC',case,attempt,met['pass_numeric'],met['failed_frames'],met['temporal']['maximum_velocity_rad_s'],flush=True)
        if met['pass_numeric'] or attempt==2:break
        indices=set(met['failed_frames'])
        indices.update((np.flatnonzero(np.any(np.abs(np.diff(q,axis=0))>4.5*dt,axis=1))+1).tolist())
        indices.update((np.flatnonzero(np.any(np.abs(np.diff(q,n=2,axis=0))>130*dt**2,axis=1))+2).tolist())
        if not indices:break
        # Separate local clusters prevents a distant defect expanding one dense
        # cone problem to the whole trajectory. Same rule for every input.
        clusters=[]
        for f in sorted(indices):
            if not clusters or f-clusters[-1][-1]>48:clusters.append([f])
            else:clusters[-1].append(f)
        fits=[]
        for cluster in clusters:
            begin=max(0,min(cluster)-24);end=min(len(q),max(cluster)+25)
            for arm in range(2):
                start=time.monotonic()
                q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,{},trust_radius=.02,max_iterations=300)
                fits.append(dict(window=[begin,end],arm=arm,fit=fit,runtime_s=time.monotonic()-start))
        atomic_npz(folder/f'CANDIDATE_{attempt}.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        atomic_json(folder/f'FIT_{attempt}.json',dict(fits=fits))
    out=folder/'FINAL_CANDIDATE.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    atomic_json(folder/'BOUNDED_RESULT.json',dict(metrics=met,trajectory=file_record(out),geometry='NOT_YET_REQUALIFIED',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1])
