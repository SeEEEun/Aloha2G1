#!/usr/bin/env python3
"""Bounded common smoothing seeds around adaptive branch flags only."""
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.qualify_train_candidate_geometry_v5 import run as qualify

def run(case,path):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=full_chain_enclosures(g);t,h,ts,source=load_input(case,g,n)
    path=Path(path).resolve();original=np.load(path)['q'].copy();folder=path.parent/(path.stem+'_common_branch_v1')
    dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    met,act,res,lb,cert,allow=qualify_numeric(s,original,t,h,ts,bounds,slack)
    flags=met['temporal']['branch_discontinuity_frames'];atomic_json(folder/'CONTRACT.json',dict(input=file_record(path),source=source,
        trigger='existing adaptive branch check',flagged_frames=flags,gaussian_seed_sigma_frames=[2,4,6],padding=24,restores=2,budget=400,aggregate_threshold_is_not_acceptance=True))
    if not flags:return
    mask=np.ones_like(original,bool)
    blend=np.zeros(len(t))
    for f in flags:
        begin=max(2,f-24);end=min(len(t)-2,f+25);mask[begin:end]=False
        u=np.linspace(0,1,end-begin);blend[begin:end]=np.maximum(blend[begin:end],np.sin(np.pi*u)**2)
    mask[:,[6,13]]=True
    for i,sigma in enumerate((2,4,6)):
        smoothed=gaussian_filter1d(original,sigma,axis=0,mode='nearest',truncate=4)
        q=original+blend[:,None]*(smoothed-original);q[mask]=original[mask]
        for attempt in range(3):
            m,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            atomic_json(folder/f'CHECK_{i}_{attempt}.json',dict(metrics=m));print('COMMON_BRANCH_CONTINUATION',case,i,attempt,m['pass_numeric'],m['failed_frames'],m['temporal']['branch_discontinuity_frames'],flush=True)
            if m['pass_numeric'] or attempt==2:break
            q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=original)
            atomic_json(folder/f'FIT_{i}_{attempt}.json',fit)
        out=folder/f'CANDIDATE_{i}.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        if m['pass_numeric']:
            qualify(case,str(out));return
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_BRANCH_RECOVERY_REQUIRED',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
