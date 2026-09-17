#!/usr/bin/env python3
"""Common geometry-derived posture prior; unchanged physical/Cartesian gates."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_model_posture_seed_v5 import model_posture_seed
from tools.common_fast_hard_witness_v5 import hard_witness

def run(case):
    oc=verified_oracle_contract()[0];g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));folder=TRAIN/case/'model_posture_candidate_v1'
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],n);slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    if (TRAIN/case/'SOURCE_POSITION_PASS.json').exists():print('REUSE_PASS',case,flush=True);return
    seed,derivation=model_posture_seed(g,n)
    atomic_json(folder/'CONTRACT.json',dict(seed=seed.tolist(),derivation=derivation,source=source,
        posture_weight=.0001,maximum_frame_evaluations=180,maximum_restores=3,restore_evaluations=400,
        no_method_or_target_input_to_seed=True,natural_q0_unchanged=True,preparation_duration_seconds=.7))
    ip=folder/'MODEL_POSTURE_GUIDE.npz'
    if ip.exists():q=np.load(ip)['q'].copy()
    else:
        q=[]
        for f in range(len(t)):
            v,fit=oracle.fit(t[f],h[f],seed,max_evaluations=180,reference=seed,posture_weight=.0001)
            v[[6,13]]=n[[6,13]];q.append(v)
            if f%100==0:print('MODEL_POSTURE_GUIDE',case,f,fit['maximum_residual_m'],flush=True)
        q=np.array(q);atomic_npz(ip,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    natural=np.array(read(INITIAL)['g1_14_arm_initial_q_rad']);prefix=preparation_path(natural,q[0],21)
    pm=temporal_metrics(prefix,dt,s.config);pg=[dict(frame=f,records=c.inspect(v,*h[0])) for f,v in enumerate(prefix)]
    pb=[r['frame'] for r in pg if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
    atomic_json(folder/'FIXED_PREPARATION_CHECK.json',dict(temporal=pm,geometry=pg,blocked_frames=pb,natural_q0_exact=bool(np.array_equal(prefix[0],natural)),duration_seconds=.7))
    if not pm['pass_temporal'] or pb:
        atomic_json(folder/'BOUNDED_RESULT.json',dict(status='BRANCH_PREPARATION_INVALID',next='OTHER_COMMON_BRANCH'));return
    mask=np.zeros_like(q,bool);mask[0]=True;mask[:,[6,13]]=True;fixed=q.copy()
    for attempt in range(4):
        qp=ip if not attempt else folder/f'RESTORED_{attempt-1}.npz'
        if attempt and qp.exists():q=np.load(qp)['q'].copy()
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=met,trajectory=file_record(qp)))
        print('MODEL_POSTURE_NUMERIC',case,attempt,met['pass_numeric'],len(met['failed_frames']),met['temporal']['maximum_velocity_rad_s'],flush=True)
        if met['pass_numeric'] or attempt==3:break
        out=folder/f'RESTORED_{attempt}.npz'
        if not out.exists():
            start=time.monotonic();q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=fixed)
            atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts);atomic_json(folder/f'FIT_{attempt}.json',dict(fit=fit,runtime_s=time.monotonic()-start))
    if not met['pass_numeric']:
        atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_POSITION_RECOVERY_REQUIRED',metrics=met));return
    probe=[]
    for f,(v,hh) in enumerate(zip(q,h)):
        rr=c.proxy._records(v,*hh);probe.append((max((x['penetration_depth_m'] for x in rr),default=0),f))
    for depth,f in sorted(probe,reverse=True)[:8]:
        if depth<=0:continue
        try:witness=hard_witness(c,q[f],h[f])
        except Exception:witness=None
        if witness:
            atomic_json(folder/'HARD_GEOMETRY_REJECTION.json',dict(frame=f,witness=witness,trajectory=file_record(qp)));return
    geometry=[]
    for f,(v,hh) in enumerate(zip(q,h)):
        p=folder/'frame_geometry'/f'FRAME_{f:04d}.json'
        if p.exists():r=read(p)
        else:r=dict(frame=f,records=c.inspect(v,*hh));atomic_json(p,r)
        geometry.append(r)
    gp=folder/'GEOMETRY.json';atomic_json(gp,dict(geometry=geometry,trajectory=file_record(qp)))
    bad=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
    full=np.vstack((prefix[:-1],q));tm=temporal_metrics(full,dt,s.config)
    out=folder/'FINAL_CANDIDATE.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
        EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
    record=dict(case=case,metrics=met,trajectory=file_record(out),geometry=file_record(gp),blocked_frames=bad,
        preparation_blocked_frames=pb,complete_temporal=tm,source=source,preparation_seconds=.7,contract=file_record(folder/'CONTRACT.json'))
    atomic_json(folder/'FINAL_QUALIFICATION.json',record)
    print('MODEL_POSTURE_FINAL',case,'blocked',bad,'temporal',tm['pass_temporal'],flush=True)
    if not bad and tm['pass_temporal']:atomic_json(TRAIN/case/'SOURCE_POSITION_PASS.json',record)

if __name__=='__main__':run(sys.argv[1])
