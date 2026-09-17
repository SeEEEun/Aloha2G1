#!/usr/bin/env python3
"""Finite common source-trajectory candidates; no acceptance shortcuts."""
from pathlib import Path
import sys,time,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_master_autonomous import MASTER,RUN,QUAL,INITIAL,read,file_record,atomic_json,atomic_npz,log
from tools.cartesian_reachability_forensic import model,BASELINE,DEST
from tools.master_autonomous_provenance import verified_oracle_contract
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.final_common_position_solver import CommonPositionSolver
from tools.common_g1_position_bounds import outer_enclosures,residual_lower_bounds
from tools.master_autonomous_common import propagated_candidate,refine_trajectory,temporal_metrics


def kinematic_metrics(solver,q,target,hands,dt,cfg,balls):
    actual=np.array([solver.pose_jacobian(v,h)[0] for v,h in zip(q,hands)])
    residual=np.linalg.norm(actual-target,axis=2)
    lower=np.array([residual_lower_bounds(t,balls) for t in target])
    certified=np.max(lower,axis=1)>.01000001
    # Screening uses ONLY the user's absolute 2mm slack cap. This is NOT the
    # repeatability-derived final slack and cannot promote a final trajectory.
    screen=np.where(certified,np.max(lower,axis=1)+.002,.01)
    maximum=np.max(residual,axis=1)
    return dict(temporal=temporal_metrics(q,dt,cfg),raw_acceptance=float(np.mean(maximum<=.01)),
        certified_frames=int(certified.sum()),screen_failures=np.flatnonzero(maximum>screen).tolist(),
        maximum_residual_mm=float(maximum.max()*1000),mean_residual_mm=float(maximum.mean()*1000),
        finite=bool(np.isfinite(q).all()),hard_limit_violations=int(np.count_nonzero((q<solver.g1.arm_limits[:,0])|(q>solver.g1.arm_limits[:,1])))),actual,residual,lower


def run(case):
    cfg=read(QUAL);oc=verified_oracle_contract()[0];g1,collision,natural=model()
    oracle=FramewiseReachabilityOracle(g1,collision,oc['oracle'],natural)
    solver=CommonPositionSolver(g1,collision,cfg,natural);g1.assign(natural);balls=outer_enclosures(g1)
    source=BASELINE/f'{case}.npz'
    with np.load(source) as z:target=z['RAW_REPRESENTATION_TARGET'].copy();hands=z['common_hand_q'].copy();timestamps=z['source_timestamp'].copy();baseline=z['q'].copy()
    dt=float(np.median(np.diff(timestamps)));folder=RUN/case;results=[]
    startup=np.array(read(RUN/'startup'/f'{case}.json')['first_task_q'])
    seeds=[('forward_natural',natural,False),('backward_natural',natural,True),
           ('forward_startup',startup,False),('backward_standing',oracle.seeds[1],True)]
    for label,seed,reverse in seeds:
        path=folder/f'{label}.npz';metric_path=folder/f'{label}.json'
        if path.exists():
            with np.load(path) as z:q=z['q'].copy()
        else:
            started=time.monotonic();q=propagated_candidate(oracle,target,hands,seed,reverse)
            atomic_npz(path,q=q,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=timestamps)
            print('PROPAGATED',case,label,round(time.monotonic()-started,2),flush=True)
        metrics,actual,residual,lower=kinematic_metrics(solver,q,target,hands,dt,cfg,balls)
        atomic_json(metric_path,dict(case=case,candidate=label,source=file_record(source),trajectory=file_record(path),**metrics))
        print(case,label,metrics,flush=True);results.append((label,q,metrics))
    # Refine two independently propagated branches and the old physically valid
    # baseline. Same deterministic candidate family for all input trajectories.
    for label,q0 in [(results[0][0],results[0][1]),(results[1][0],results[1][1]),('baseline',baseline)]:
        for weight in (.003,.001,.0003):
            tag=f'{label}_sparse_{weight}';path=folder/f'{tag}.npz';mp=folder/f'{tag}.json'
            if path.exists():
                with np.load(path) as z:q=z['q'].copy()
                fit=read(mp).get('fit',{})
            else:
                print('REFINEMENT_START',case,tag,flush=True);started=time.monotonic()
                q,fit=refine_trajectory(solver,target,hands,q0,weight,100)
                fit['runtime_s']=time.monotonic()-started
                atomic_npz(path,q=q,RAW_REPRESENTATION_TARGET=target,actual_wrist_position_model=np.array([solver.pose_jacobian(v,h)[0] for v,h in zip(q,hands)]),common_hand_q=hands,source_timestamp=timestamps)
            metrics,actual,residual,lower=kinematic_metrics(solver,q,target,hands,dt,cfg,balls)
            atomic_json(mp,dict(case=case,candidate=tag,source=file_record(source),trajectory=file_record(path),fit=fit,**metrics))
            print(case,tag,metrics,flush=True);results.append((tag,q,metrics))
    # Exhaustive detailed geometry only for the best kinematic candidate; all
    # rejected candidates stay persisted. Geometry failures cannot be waived.
    label,q,metrics=min(results,key=lambda r:(len(r[2]['screen_failures']),not r[2]['temporal']['pass_temporal'],r[2]['temporal']['maximum_step_norm_rad'],r[2]['mean_residual_mm']))
    geometry=[]
    for f,(v,h) in enumerate(zip(q,hands)):
        rr=collision.inspect(v,*h);geometry.append(dict(frame=f,records=rr))
        if f%100==0:print('GEOMETRY',case,label,f,flush=True)
    blocked=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
    result=dict(case=case,selected_candidate=label,metrics=metrics,blocked_frames=blocked,geometry=geometry,
        qualified=False,note='Candidate search; final repeatability slack, unresolved classification, preparation join and physical checks remain mandatory.')
    atomic_json(folder/'BOUNDED_CANDIDATE_RESULT.json',result)
    print('BOUNDED_CANDIDATE_COMPLETE',case,'blocked',len(blocked),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();run(a.case)
