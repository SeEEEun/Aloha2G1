#!/usr/bin/env python3
"""Common hard-temporal local refinement of any saved candidate (I/O only)."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_conic_trust_realization_v6 import solve
from tools.qualify_train_candidate_geometry_v5 import run as qualify

def run(case,path):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=full_chain_enclosures(g);t,h,ts,source=load_input(case,g,n)
    path=Path(path).resolve();q=np.load(path)['q'].copy();folder=path.parent/(path.stem+'_common_conic_v2')
    dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(path),source=source,attempts=2,window_padding=24,iterations=300,trust_radius=.15,acceptance_unchanged=True))
    for attempt in range(3):
        m,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=m));print('CANDIDATE_CONIC',case,attempt,m['pass_numeric'],m['failed_frames'],flush=True)
        if m['pass_numeric'] or attempt==2:break
        for arm in range(2):
            ids=slice(7*arm,7*arm+7)
            bad=set(np.flatnonzero(res[:,arm]>allow[:,arm]).tolist())
            bad.update((np.flatnonzero(np.any(np.abs(np.diff(q[:,ids],axis=0))>4.5*dt,axis=1))+1).tolist())
            bad.update((np.flatnonzero(np.any(np.abs(np.diff(q[:,ids],n=2,axis=0))>130*dt**2,axis=1))+2).tolist())
            clusters=[]
            for f in sorted(bad):
                if not clusters or f-clusters[-1][-1]>48:clusters.append([f])
                else:clusters[-1].append(f)
            for ci,cluster in enumerate(clusters):
                begin=max(0,min(cluster)-24);end=min(len(q),max(cluster)+25)
                q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,{},trust_radius=.15,max_iterations=300)
                atomic_json(folder/f'FIT_{attempt}_{arm}_{ci}.json',dict(fit=fit,window=[begin,end]))
        atomic_npz(folder/f'CANDIDATE_{attempt}.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    out=folder/'FINAL_CANDIDATE.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    atomic_json(folder/'NUMERIC_RESULT.json',dict(metrics=m,trajectory=file_record(out)))
    if m['pass_numeric']:qualify(case,str(out))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])

