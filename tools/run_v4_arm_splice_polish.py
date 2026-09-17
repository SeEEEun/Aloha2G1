#!/usr/bin/env python3
"""Correct untouched-DOF splice, then bounded common numerical refinement."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric,restore

def run(case):
    oc=verified_oracle_contract()[0];g,c,n=model();s=CommonPositionSolver(g,c,read(QUAL),n);g.assign(n);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)));slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    old=V4/case/'hard_physical_geometry_window_v1';contract=read(old/'CONTRACT.json');base=np.load(Path(contract['source']['path']))['q'].copy();ids=np.concatenate([np.arange(a*7,a*7+6) for a in contract['arms']])
    folder=V4/case/'untouched_dof_splice_fix_v1';ap=RUN/case/'certified_anchor_recovery/CERTIFIED_ANCHORS.npz';a=np.load(ap);anchors=a['q'];mask=a['fixed_mask']
    atomic_json(folder/'CONTRACT.json',dict(reason='Original hard-window runner copied14-DOF alternative seed but optimized only influenced arm6DOF; the unoptimized opposite arm could splice discontinuously. Preserve base trajectory for all unoptimized DOFs.',
        optimized_indices=ids.tolist(),source=file_record(Path(contract['source']['path'])),target_and_physical_limits_unchanged=True,
        maximum_candidates=3,maximum_polish_attempts=2,maximum_evaluations=400,implementation=file_record(Path(__file__))))
    for k in range(3):
        sub=folder/f'SEED_{k}';src=old/f'CANDIDATE_{k}.npz';q=base.copy();q[:,ids]=np.load(src)['q'][:,ids]
        atomic_npz(sub/'SPLICE_CORRECTED.npz',q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        for attempt in range(3):
            p=sub/f'POLISHED_{attempt-1}.npz'
            if attempt and p.exists():q=np.load(p)['q'].copy()
            met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            atomic_json(sub/f'NUMERIC_{attempt}.json',dict(metrics=met))
            print('SPLICE_POLISH_NUMERIC',case,k,attempt,met['pass_numeric'],met['failed_frames'],met['temporal']['maximum_velocity_rad_s'],flush=True)
            if met['pass_numeric']:
                rankpath=sub/f'RANKING_{attempt}.json'
                if rankpath.exists():rank=read(rankpath)['rows']
                else:
                    rank=rank_constant_nullspace(g,c,q,h);atomic_json(rankpath,dict(rows=rank))
                for j,row in enumerate(rank[:4]):
                    qv=q.copy();qv[:,[6,13]]=row['wrist_joint_values'];gp=sub/f'GEOMETRY_{attempt}_{j}.json'
                    if gp.exists():geom=read(gp)['geometry']
                    else:
                        geom=[dict(frame=f,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(qv,h))];atomic_json(gp,dict(geometry=geom))
                    blocked=[r['frame'] for r in geom if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
                    print('SPLICE_POLISH_GEOMETRY',case,k,attempt,j,'blocked',len(blocked),flush=True)
                    if not blocked:
                        met,act,res,lb,cert,allow=qualify_numeric(s,qv,t,h,ts,bounds,slack);assert met['pass_numeric'];out=sub/'QUALIFIED_SOURCE_Q.npz'
                        atomic_npz(out,q=qv,EXECUTABLE_Q=qv,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,EXECUTABLE_FK_POSITION=act,
                            POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
                        atomic_json(V4/case/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(out),geometry=file_record(gp),source=source,acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'));return
                break
            if attempt==2:break
            out=sub/f'POLISHED_{attempt}.npz';mp=sub/f'POLISHED_{attempt}.json'
            if not out.exists():
                start=time.monotonic();q,fit=restore(s,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=anchors)
                atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts);atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start))
    atomic_json(folder/'COMPLETE.json',dict(status='BOUNDED_SPLICE_REPAIR_AND_POLISH_COMPLETE',global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
