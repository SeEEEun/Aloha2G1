#!/usr/bin/env python3
"""Identical finite elastic-conic continuation for every failed input."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_sum_conic_realization_v5 import solve
from tools.qualify_train_candidate_geometry_v5 import run as qualify

def run(case,path):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    t,h,ts,source=load_input(case,g,n);path=Path(path).resolve();q=np.load(path)['q'].copy();dt=float(np.median(np.diff(ts)))
    folder=path.parent/(path.stem+'_sum_conic_v1');slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(path),source=source,padding=[48,96],maximum_iterations=500,trust_radius=.15,
        elastic_variables='internal objective only, zero added final acceptance slack',acceptance_unchanged=True))
    for attempt,padding in enumerate((48,96)):
        m,a,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=m));print('SUM_CONIC_CHECK',case,attempt,m['pass_numeric'],m['failed_frames'],flush=True)
        if m['pass_numeric']:break
        for arm in range(2):
            ids=slice(arm*7,arm*7+7);bad=set(np.flatnonzero(res[:,arm]>allow[:,arm]).tolist())
            bad.update((np.flatnonzero(np.any(np.abs(np.diff(q[:,ids],axis=0))>4.5*dt,axis=1))+1).tolist())
            bad.update((np.flatnonzero(np.any(np.abs(np.diff(q[:,ids],n=2,axis=0))>130*dt**2,axis=1))+2).tolist())
            if not bad:continue
            clusters=[]
            for f in sorted(bad):
                if not clusters or f-clusters[-1][-1]>2*padding:clusters.append([f])
                else:clusters[-1].append(f)
            for ci,cluster in enumerate(clusters):
                begin=max(0,min(cluster)-padding);end=min(len(q),max(cluster)+padding+1)
                q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,max_iterations=500)
                atomic_json(folder/f'FIT_{attempt}_{arm}_{ci}.json',dict(fit=fit,window=[begin,end]))
        atomic_npz(folder/f'CANDIDATE_{attempt}.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    m,a,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack);out=folder/'FINAL_CANDIDATE.npz'
    atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    atomic_json(folder/'FINAL_NUMERIC.json',dict(metrics=m,trajectory=file_record(out)))
    if m['pass_numeric']:qualify(case,str(out))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
