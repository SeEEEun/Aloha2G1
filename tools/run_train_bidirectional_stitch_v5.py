#!/usr/bin/env python3
"""Common bounded forward/backward posture-guide continuation.

A guide splice is only an optimizer seed. Raw targets, timestamps and commands
are unchanged; complete original acceptance follows every candidate.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.run_train_conic_from_candidate_v6 import run as conic_refine

def run(case,guide_folder):
    oc=verified_oracle_contract()[0];g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    guide_folder=Path(guide_folder).resolve();paths=[guide_folder/d/'GEOMETRY_AWARE_GUIDE.npz' for d in ('forward','reverse')]
    F,R=[np.load(p)['q'].copy() for p in paths];assert np.isfinite(F).all() and np.isfinite(R).all()
    reports=[read(guide_folder/d/'GUIDE_REPORT.json')['frames'] for d in ('forward','reverse')]
    bad=[]
    for rows in reports:
        mask=np.ones(len(t),int)
        for row in rows:mask[row['frame']]=int(not row['selected']['geometry_valid'])
        bad.append(mask)
    ranks=[]
    for f in range(25,len(t)-25):
        delta=R[f]-F[f-1];count=int(bad[0][:f].sum()+bad[1][f:].sum())
        ranks.append(dict(frame=f,known_geometry_invalid=count,splice_velocity_ratio=float(np.max(np.abs(delta))/(4.5*dt)),joint_gap=float(np.linalg.norm(delta))))
    ranks.sort(key=lambda r:(r['known_geometry_invalid'],r['splice_velocity_ratio'],r['joint_gap'],r['frame']))
    selected=[]
    for row in ranks:
        if all(abs(row['frame']-r['frame'])>=24 for r in selected):selected.append(row)
        if len(selected)==3:break
    folder=guide_folder/'common_bidirectional_stitch_v1';atomic_json(folder/'CONTRACT.json',dict(inputs=[file_record(p) for p in paths],source=source,
        candidates=selected,splice_blend_half_width=24,restores=2,restore_budget=400,final_acceptance_unchanged=True))
    for ci,row in enumerate(selected):
        f=row['frame'];u=np.clip((np.arange(len(t))-(f-24))/48,0,1);w=u**3*(10-15*u+6*u*u)
        q=(1-w[:,None])*F+w[:,None]*R;q[0]=F[0];mask=np.zeros_like(q,bool);mask[0]=True;mask[:,[6,13]]=True
        for attempt in range(3):
            m,a,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            atomic_json(folder/f'CHECK_{ci}_{attempt}.json',dict(metrics=m,splice=row));print('COMMON_BIDIRECTIONAL_STITCH',case,ci,f,attempt,m['pass_numeric'],m['failed_frames'],m['temporal']['maximum_velocity_rad_s'],flush=True)
            if m['pass_numeric'] or attempt==2:break
            q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=q)
            atomic_json(folder/f'FIT_{ci}_{attempt}.json',fit)
        out=folder/f'CANDIDATE_{ci}.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        conic_refine(case,str(out))
        if (TRAIN/case/'SOURCE_POSITION_PASS.json').exists():return
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_BIDIRECTIONAL_CANDIDATES_COMPLETE',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
