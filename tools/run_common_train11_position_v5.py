#!/usr/bin/env python3
"""Common fixed TRAIN qualification builder; method labels only locate inputs."""
from pathlib import Path
import sys,time,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_full_chain_certificate_v5 import full_chain_enclosures
from tools.common_physical_position_v4 import restore
from tools.common_robust_proxy_penalty import RobustProxyPenaltySolver
from tools.master_autonomous_common import propagated_candidate,preparation_path
from tools.common_physical_position_v4 import temporal_metrics

TRAIN=OUT/'02_common_execution_qualification/common_train11_position_v5'

def run(case):
    oc=verified_oracle_contract()[0]
    assert read(ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json')['status']=='SMOKE3_COMMON_EXECUTABLE_POSITION_QUALIFIED'
    split=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids']
    assert int(case.split('_EP')[1]) in split
    g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],n)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));folder=TRAIN/case
    if (folder/'SOURCE_POSITION_PASS.json').exists():print('REUSE_TRAIN11_PASS',case,flush=True);return
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    startup=np.array(read(RUN/'startup'/f'{case}.json')['first_task_q']);opened=h[0].copy()
    contract=dict(source=source,certificate=file_record(ST5/'full_chain_certificate/MODEL_ONLY_CERTIFICATE.json'),
        acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),
        families=['forward_startup','backward_natural','independent_natural'],per_family_restore_attempts=3,maximum_restore_evaluations=400,
        source_first_witness='fixed cached source-conditioned first target q, except exact terminal-wrist position nullspace',
        preparation_seconds=.7,preparation_frames=21,targets_unchanged=True,method_blind_core=True,
        implementation=file_record(Path(__file__)))
    cp=folder/'CONTRACT.json'
    if not cp.exists():atomic_json(cp,contract)
    summaries=[]
    for family in contract['families']:
        sub=folder/family;ip=sub/'INITIAL.npz'
        if ip.exists():q=np.load(ip)['q'].copy()
        else:
            print('TRAIN11_SEEDS',case,family,flush=True)
            if family=='forward_startup':q=propagated_candidate(oracle,t,h,startup,False)
            elif family=='backward_natural':q=propagated_candidate(oracle,t,h,n,True)
            else:
                q=np.array([oracle.fit(tt,hh,n,max_evaluations=180)[0] for tt,hh in zip(t,h)])
            q[0]=startup;q[:,[6,13]]=n[[6,13]]
            atomic_npz(ip,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        mask=np.zeros_like(q,bool);mask[0,:]=True;mask[:,[6,13]]=True;fixed=q.copy()
        for attempt in range(4):
            qp=ip if not attempt else sub/f'RESTORED_{attempt-1}.npz'
            if attempt and qp.exists():q=np.load(qp)['q'].copy()
            met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            mp=sub/f'CHECK_{attempt}.json';atomic_json(mp,dict(metrics=met,trajectory=file_record(qp)))
            print('TRAIN11_NUMERIC',case,family,attempt,met['pass_numeric'],'failed',len(met['failed_frames']),
                'velocity',met['temporal']['maximum_velocity_rad_s'],'acceleration',met['temporal']['maximum_acceleration_rad_s2'],flush=True)
            if met['pass_numeric']:break
            if attempt==3:break
            out=sub/f'RESTORED_{attempt}.npz';fp=sub/f'FIT_{attempt}.json'
            if not out.exists():
                start=time.monotonic();q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=fixed)
                atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
                atomic_json(fp,dict(fit=fit,runtime_s=time.monotonic()-start))
        if not met['pass_numeric']:
            summaries.append(dict(family=family,metrics=met,trajectory=file_record(qp),geometry_not_qualified=True));continue
        ranking=sub/'NULLSPACE_RANKING.json'
        if ranking.exists():rank=read(ranking)['rows']
        else:rank=rank_constant_nullspace(g,c,q,h);atomic_json(ranking,dict(rows=rank,trajectory=file_record(qp)))
        for index,row in enumerate(rank[:4]):
            v=q.copy();v[:,[6,13]]=row['wrist_joint_values']
            met,act,res,lb,cert,allow=qualify_numeric(s,v,t,h,ts,bounds,slack)
            gp=sub/f'GEOMETRY_{index}.json';vp=sub/f'NULLSPACE_{index}.npz'
            if gp.exists():geometry=read(gp)['geometry']
            else:
                geometry=[dict(frame=f,records=c.inspect(a,*hh)) for f,(a,hh) in enumerate(zip(v,h))]
                atomic_json(gp,dict(geometry=geometry))
            bad=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
            prefix=preparation_path(np.array(read(INITIAL)['g1_14_arm_initial_q_rad']),v[0],21)
            full=np.vstack((prefix[:-1],v));temporal=temporal_metrics(full,dt,s.config)
            prefix_geom=[dict(frame=f,records=c.inspect(a,*opened)) for f,a in enumerate(prefix[:-1])]
            prep_bad=[r['frame'] for r in prefix_geom if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
            atomic_npz(vp,q=v,EXECUTABLE_Q=v,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
                EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
            record=dict(case=case,family=family,index=index,metrics=met,trajectory=file_record(vp),geometry=file_record(gp),
                blocked_frames=bad,preparation_blocked_frames=prep_bad,preparation_geometry=prefix_geom,complete_temporal=temporal,
                source=source,contract=file_record(cp))
            atomic_json(sub/f'QUALIFICATION_{index}.json',record);summaries.append(record)
            print('TRAIN11_GEOMETRY',case,family,index,'blocked',len(bad),'prep',len(prep_bad),'temporal',temporal['pass_temporal'],flush=True)
            if met['pass_numeric'] and not bad and not prep_bad and temporal['pass_temporal']:
                atomic_json(folder/'SOURCE_POSITION_PASS.json',record);return
    atomic_json(folder/'BOUNDED_CANDIDATE_RESULT.json',dict(case=case,candidates=summaries,status='COMMON_TRAIN_POSITION_RECOVERY_REQUIRED',
        next='COMMON_ANCHOR_OR_LOCAL_GEOMETRY_RECOVERY',global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1])
