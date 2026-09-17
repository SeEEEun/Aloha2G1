#!/usr/bin/env python3
"""Common bounded local bidirectional continuation from framewise anchors."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_model_posture_seed_v5 import model_posture_seed
from tools.qualify_train_candidate_geometry_v5 import run as qualify

def run(case,path):
    oc=verified_oracle_contract()[0];g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],n);g.assign(n);bounds=full_chain_enclosures(g)
    t,h,ts,source=load_input(case,g,n);path=Path(path).resolve();original=np.load(path)['q'].copy()
    dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=path.parent/(path.stem+'_local_anchor_v1');m,a,res,lb,cert,allow=qualify_numeric(s,original,t,h,ts,bounds,slack)
    bad=m['failed_frames'];assert bad
    # Peak certified morphology demand and peak missed Cartesian residual are
    # geometry/numerical anchors, never source task-success or object waypoints.
    anchors=sorted(set([max(bad,key=lambda f:float(lb[f].max())),max(bad,key=lambda f:float(np.max(res[f]-allow[f])))]))
    bank=np.load(ST5/'common_training_seed_bank/MODEL_Q_SEEDS.npz')
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(path),source=source,anchors=anchors,padding=[48,96],
        anchor_seeds=10,frame_fit_budget=180,restore_budget=400,restores=2,raw_targets_changed=False,event_times_changed=False))
    for ai,anchor in enumerate(anchors):
        distances=np.linalg.norm(bank['model_wrist_position'].reshape(len(bank['q']),6)-t[anchor].reshape(1,6),axis=1)
        seeds=[original[anchor],s.natural,*bank['q'][np.argsort(distances,kind='stable')[:8]]]
        witnesses=[]
        for sid,seed in enumerate(seeds):
            value,fit=oracle.fit(t[anchor],h[anchor],seed,max_evaluations=180,reference=seed,posture_weight=.0001)
            value[[6,13]]=original[anchor,[6,13]];err=np.linalg.norm(s.pose_jacobian(value,h[anchor])[0]-t[anchor],axis=1)
            witnesses.append((float(np.max(np.maximum(err-allow[anchor],0))),float(np.linalg.norm(value-original[anchor])),sid,value,err,fit))
        best=min(witnesses,key=lambda r:r[:3]);atomic_json(folder/f'ANCHOR_{anchor}.json',dict(selected_seed=best[2],q=best[3].tolist(),residual_m=best[4].tolist(),
            candidates=[dict(seed=r[2],q=r[3].tolist(),residual_m=r[4].tolist(),fit=r[5]) for r in witnesses]))
        for wi,padding in enumerate((48,96)):
            begin=max(2,min(bad+[anchor])-padding);end=min(len(t)-2,max(bad+[anchor])+padding+1)
            q=original.copy();q[anchor]=best[3]
            for order in (range(anchor-1,begin-1,-1),range(anchor+1,end)):
                previous=q[anchor]
                for f in order:
                    value,fit=oracle.fit(t[f],h[f],previous,max_evaluations=180,reference=previous,posture_weight=.0001)
                    value[[6,13]]=original[f,[6,13]];q[f]=value;previous=value
            mask=np.ones_like(q,bool);mask[begin:end]=False;mask[:,[6,13]]=True
            for attempt in range(3):
                m,a,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
                atomic_json(folder/f'CHECK_{ai}_{wi}_{attempt}.json',dict(metrics=m));print('LOCAL_FRAMEWISE_ANCHOR',case,anchor,padding,attempt,m['pass_numeric'],m['failed_frames'],m['temporal']['maximum_velocity_rad_s'],flush=True)
                if m['pass_numeric'] or attempt==2:break
                q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=original)
                atomic_json(folder/f'FIT_{ai}_{wi}_{attempt}.json',fit)
            out=folder/f'CANDIDATE_{ai}_{wi}.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
            if m['pass_numeric']:qualify(case,str(out));return
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_LOCAL_ANCHOR_RECOVERY_REQUIRED',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
