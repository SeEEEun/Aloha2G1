#!/usr/bin/env python3
"""Common bounded analytic redundancy bridge to a feasible future anchor.

Conditional joint boxes are search constraints derived from unchanged physical
velocity limits. Failure proves neither global nor framewise infeasibility.
"""
from pathlib import Path
import sys,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_analytic_position_branches_v5 import AnalyticPositionBranches
from tools.run_train_conic_from_candidate_v6 import run as refine

def run(case,path):
    oc=verified_oracle_contract()[0];g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);analytic=AnalyticPositionBranches(g)
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],n);g.assign(n);bounds=full_chain_enclosures(g)
    t,h,ts,source=load_input(case,g,n);path=Path(path).resolve();original=np.load(path)['q'].copy();dt=float(np.median(np.diff(ts)))
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];m,a,res,lb,cert,allow=qualify_numeric(s,original,t,h,ts,bounds,slack)
    future=min(f for f in m['failed_frames'] if np.max(lb[f])>.01000001);arm=int(np.argmax(lb[future]));base=arm*7
    folder=path.parent/(path.stem+'_analytic_bridge_v1');bank=np.load(ST5/'common_training_seed_bank/MODEL_Q_SEEDS.npz')
    near=np.argsort(np.linalg.norm(bank['model_wrist_position'].reshape(len(bank['q']),6)-t[future].reshape(1,6),axis=1),kind='stable')[:8]
    best=[]
    for sid,seed in enumerate([original[future],n,*bank['q'][near]]):
        q,fit=oracle.fit(t[future],h[future],seed,max_evaluations=300,reference=seed,posture_weight=.0001)
        q[(1-arm)*7:(2-arm)*7]=original[future,(1-arm)*7:(2-arm)*7];q[[6,13]]=original[future,[6,13]]
        error=np.linalg.norm(s.pose_jacobian(q,h[future])[0]-t[future],axis=1)
        if np.all(error<=allow[future]):
            rr=c.inspect(q,*h[future]);valid=not any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in rr)
            if valid:best.append((float(np.linalg.norm(q-original[future])),sid,q,error))
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(path),source=source,future_frame=future,arm=arm,lookback_frames=[5,10,20],
        future_seeds=10,shoulder_grid_per_axis=17,geometry_candidate_budget=16,physical_limits_unchanged=True,source_targets_unchanged=True,
        no_witness_is_not_global_infeasibility=True,analytic_validation=file_record(ST5/'analytic_branch_audit/MODEL_ONLY_ROUNDTRIP.json')))
    if not best:
        atomic_json(folder/'BOUNDED_RESULT.json',dict(status='FUTURE_ANCHOR_NOT_WITNESSED'));return
    _,sid,fq,fe=min(best,key=lambda r:r[:2]);atomic_json(folder/'FUTURE_ANCHOR.json',dict(q=fq.tolist(),seed=sid,residual_m=fe.tolist()))
    for lookback in (5,10,20):
        before=future-lookback
        if before<2:continue
        lo=np.maximum(s.lower,fq-4.5*dt*lookback);hi=np.minimum(s.upper,fq+4.5*dt*lookback)
        grids=[np.linspace(lo[k]+1e-10,hi[k]-1e-10,17) for k in range(base,base+3)];pool=[];count=0;minimum=np.inf
        for shoulders in itertools.product(*grids):
            for row in analytic.candidates(t[before,arm],shoulders,original[before],h[before],arm,lo,hi,allow_closest=True):
                count+=1;minimum=min(minimum,row['residual_m'])
                if row['residual_m']<=allow[before,arm]:
                    value=row['q'];key=float(np.linalg.norm(value-original[before]))+float(np.linalg.norm(value-fq))
                    pool.append((key,value,row['residual_m']))
        pool.sort(key=lambda r:r[0]);witness=None;tested=[]
        for key,value,error in pool[:16]:
            rr=c.inspect(value,*h[before]);valid=not any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in rr)
            tested.append(dict(q=value.tolist(),residual_m=error,geometry=rr,valid=valid))
            if valid:witness=value;break
        atomic_json(folder/f'BRIDGE_ORACLE_{lookback}.json',dict(before=before,future=future,generated=count,numerically_eligible=len(pool),
            minimum_sampled_residual_m=None if not np.isfinite(minimum) else minimum,conditional_lower=lo.tolist(),conditional_upper=hi.tolist(),
            tested=tested,witness_found=witness is not None,global_infeasibility_proven=False))
        print('ANALYTIC_BRIDGE_ORACLE',case,lookback,count,len(pool),minimum,witness is not None,flush=True)
        if witness is None:continue
        q=original.copy();alpha=np.linspace(0,1,lookback+1)
        q[before:future+1]=(1-alpha[:,None])*witness+alpha[:,None]*fq
        begin=max(2,before-48);end=min(len(q)-2,future+49);mask=np.ones_like(q,bool);mask[begin:end]=False;mask[:,[6,13]]=True
        for attempt in range(2):
            q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=original)
            atomic_json(folder/f'FIT_{lookback}_{attempt}.json',fit)
        out=folder/f'CANDIDATE_{lookback}.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        refine(case,str(out))
        if (TRAIN/case/'SOURCE_POSITION_PASS.json').exists():return
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_ANALYTIC_BRIDGE_SEARCH_COMPLETE',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
