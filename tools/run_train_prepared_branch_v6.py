#!/usr/bin/env python3
"""Common prep-constrained alternate SOURCE_TASK_START position branch.

GLOBAL q0, prefix duration, raw targets and source event timing never change.
The first executable source-task q is selected from model-derived redundant
branches that satisfy the same unchanged prefix and source tracking contract.
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
    future,arm=np.unravel_index(np.argmax(lb),lb.shape);base=arm*7
    global_n=np.array(read(INITIAL)['g1_14_arm_initial_q_rad']);folder=path.parent/(path.stem+'_prepared_branch_v2');bank=np.load(ST5/'common_training_seed_bank/MODEL_Q_SEEDS.npz')
    scalar=preparation_path(np.zeros(14),np.ones(14),21)[:,0]
    velocity_coefficient=float(np.abs(np.diff(scalar)).max()/dt);acceleration_coefficient=float(np.abs(np.diff(scalar,n=2)).max()/dt**2)
    displacement=min(4.5/velocity_coefficient,130/acceleration_coefficient)
    lo=np.maximum(s.lower,global_n-displacement);hi=np.minimum(s.upper,global_n+displacement)
    initial=original[0].copy();initial[[6,13]]=global_n[[6,13]]
    near=np.argsort(np.linalg.norm(bank['model_wrist_position'].reshape(len(bank['q']),6)-t[future].reshape(1,6),axis=1),kind='stable')[:8]
    future_candidates=[]
    for sid,seed in enumerate([original[future],n,*bank['q'][near]]):
        value,fit=oracle.fit(t[future],h[future],seed,max_evaluations=180,reference=seed,posture_weight=.0001)
        error=np.linalg.norm(s.pose_jacobian(value,h[future])[0]-t[future],axis=1)
        future_candidates.append((float(np.maximum(error-allow[future],0).max()),float(np.linalg.norm(value-original[future])),sid,value))
    fq=min(future_candidates,key=lambda r:r[:3])[3]
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(path),source=source,global_natural_q0=file_record(INITIAL),
        prep_seconds=.7,prep_frames=21,prep_velocity_coefficient=velocity_coefficient,prep_acceleration_coefficient=acceleration_coefficient,
        prep_derived_per_joint_displacement_bound_rad=displacement,source_first_pose='alternate redundant q; incoming first Cartesian target unchanged',
        future_model_demand_frame=int(future),arm=int(arm),shoulder_grid=17,geometry_trials=24,maximum_trajectory_candidates=3,
        source_events_changed=False,registration_changed=False,raw_targets_changed=False,
        search_fix='authoritative INITIAL q0 for prefix, distinct from model-only solver prior',
        recovery_seeds_when_raw_gate_missed=10))
    pool=[]
    grids=[np.linspace(lo[k]+1e-9,hi[k]-1e-9,17) for k in range(base,base+3)]
    for shoulder in itertools.product(*grids):
        for row in analytic.candidates(t[0,arm],shoulder,initial,h[0],arm,lo,hi,allow_closest=True):
            if row['residual_m']>allow[0,arm]:continue
            value=row['q'];key=(float(np.linalg.norm(value[base:base+6]-fq[base:base+6])),float(np.linalg.norm(value-global_n)))
            pool.append((key,value,row['residual_m']))
    pool.sort(key=lambda r:r[0]);atomic_json(folder/'NUMERIC_ENDPOINTS.json',dict(count=len(pool),top=[dict(key=r[0],q=r[1].tolist(),residual_m=r[2]) for r in pool[:24]]))
    endpoints=[]
    for index,(key,value,error) in enumerate(pool[:24]):
        prefix=preparation_path(global_n,value,21);assert np.array_equal(prefix[0],global_n);tm=temporal_metrics(prefix,dt,s.config);geometry=[dict(frame=f,records=c.inspect(v,*h[0])) for f,v in enumerate(prefix)]
        blocked=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        atomic_json(folder/f'ENDPOINT_{index}.json',dict(q=value.tolist(),temporal=tm,geometry=geometry,blocked=blocked,first_target_residual_m=error))
        print('COMMON_PREPARED_BRANCH',case,index,'blocked',blocked,'temporal',tm['pass_temporal'],flush=True)
        if not blocked and tm['pass_temporal']:endpoints.append((index,value))
        if len(endpoints)==3:break
    for index,endpoint in endpoints:
        q=original.copy();q[0]=endpoint;previous=endpoint
        for f in range(1,len(q)):
            value,fit=oracle.fit(t[f],h[f],previous,max_evaluations=180,reference=previous,posture_weight=.0001)
            error=np.linalg.norm(s.pose_jacobian(value,h[f])[0]-t[f],axis=1)
            if np.any(error>allow[f]):
                near=np.argsort(np.linalg.norm(bank['model_wrist_position'].reshape(len(bank['q']),6)-t[f].reshape(1,6),axis=1),kind='stable')[:8]
                options=[]
                for sid,seed in enumerate([value,original[f],*bank['q'][near]]):
                    candidate,info=oracle.fit(t[f],h[f],seed,max_evaluations=180,reference=seed,posture_weight=.0001)
                    ee=np.linalg.norm(s.pose_jacobian(candidate,h[f])[0]-t[f],axis=1)
                    violation=float(np.maximum(ee-allow[f],0).max())
                    options.append((violation>0,violation if violation>0 else float(np.linalg.norm(candidate-previous)),sid,candidate))
                value=min(options,key=lambda row:row[:3])[3]
            value[[6,13]]=endpoint[[6,13]];q[f]=value;previous=value
            if f%100==0:print('PREPARED_BRANCH_PROPAGATION',case,index,f,flush=True)
        mask=np.zeros_like(q,bool);mask[0]=True;mask[:,[6,13]]=True
        for attempt in range(3):
            m,a,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            atomic_json(folder/f'CHECK_{index}_{attempt}.json',dict(metrics=m));print('PREPARED_BRANCH_NUMERIC',case,index,attempt,m['pass_numeric'],m['failed_frames'],m['temporal']['maximum_velocity_rad_s'],flush=True)
            if m['pass_numeric'] or attempt==2:break
            q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=q)
            atomic_json(folder/f'FIT_{index}_{attempt}.json',fit)
        out=folder/f'CANDIDATE_{index}.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        refine(case,str(out))
        if (TRAIN/case/'SOURCE_POSITION_PASS.json').exists():return
    atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_PREPARED_BRANCH_SEARCH_COMPLETE',eligible_endpoints=len(endpoints),global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])

