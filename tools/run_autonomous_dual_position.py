#!/usr/bin/env python3
"""Common executable position realization with transparent certified corrections."""
from pathlib import Path
import sys,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_master_autonomous import MASTER,RUN,QUAL,INITIAL,read,file_record,atomic_json,atomic_npz,log
from tools.final_single_variable_prepare import OUT,RESET
from tools.cartesian_reachability_forensic import model,BASELINE,DEST
from tools.master_autonomous_provenance import verified_oracle_contract
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.final_common_position_solver import CommonPositionSolver
from tools.common_hinge_orbit_bounds import orbit_enclosures,orbit_lower_bounds
from tools.master_autonomous_common import refine_trajectory,temporal_metrics
from tools.common_constrained_continuation import restore
from tools.common_wrist_nullspace_search import rank_constant_nullspace
from tools.run_autonomous_witness_continuation import independent_anchors
from tools.doll_handoff_retargeting.common import load_common_config,load_scene,branch_flags


def load_input(case,g1,natural):
    baseline=BASELINE/f'{case}.npz'
    if baseline.exists():
        with np.load(baseline) as z:
            return z['RAW_REPRESENTATION_TARGET'].copy(),z['common_hand_q'].copy(),z['source_timestamp'].copy(),file_record(baseline)
    mode,ep=case.split('_EP');raw=OUT/'01_registration/raw_references'/f'TRAIN_EP{int(ep):03d}.npz'
    common=load_common_config(RESET/'config/common_config.json')
    primitives=g1.derive_hand_primitives(load_scene(common),read(RESET/'config/proposed_config.json'),natural)
    with np.load(raw) as z:
        target=np.stack([z[f'{mode}_{s}_wrist_position_model'] for s in ('left','right')],axis=1)
        hands=np.stack([np.array(primitives['states'][s]['OPEN'])+z[f'common_{s}_close_fraction'][:,None]*(np.array(primitives['states'][s]['GRASP'])-np.array(primitives['states'][s]['OPEN'])) for s in ('left','right')],axis=1)
        ts=z['source_timestamp'].copy()
    return target,hands,ts,file_record(raw)


def qualify_numeric(solver,q,target,hands,ts,bounds,slack):
    actual=np.array([solver.pose_jacobian(v,h)[0] for v,h in zip(q,hands)])
    residual=np.linalg.norm(actual-target,axis=2);lower=np.array([orbit_lower_bounds(t,bounds) for t in target])
    cert=lower.max(axis=1)>.01000001
    allowance=np.full_like(lower,.01);allowance[cert]=np.maximum(lower[cert]+slack,.01)
    failed=np.flatnonzero(np.any(residual>allowance,axis=1)).tolist()
    temporal=temporal_metrics(q,float(np.median(np.diff(ts))),solver.config)
    branches=int(np.count_nonzero(branch_flags(q,solver.config['branch_absolute_step_norm_rad'],solver.config['branch_local_multiplier'])))
    limits=int(np.count_nonzero((q<solver.g1.arm_limits[:,0])|(q>solver.g1.arm_limits[:,1])))
    maximum=residual.max(axis=1)*1000
    result=dict(failed_frames=failed,raw_acceptance=float(np.mean(maximum<=10)),certified_unreachable_frames=np.flatnonzero(cert).tolist(),
        temporal=temporal,branch_discontinuities=branches,hard_limit_violations=limits,finite=bool(np.isfinite(q).all()),
        correction_mm=dict(mean=float(maximum.mean()),p95=float(np.quantile(maximum,.95)),max=float(maximum.max())),
        numerical_slack_mm=slack*1000,certified_optimality_gap_max_mm=float(np.max(residual[cert]-lower[cert])*1000) if np.any(cert) else None,
        certified_optimality_gap_scope='per wrist; the other reachable wrist in a certified bilateral frame retains its 10mm raw gate')
    result['pass_numeric']=bool(not failed and temporal['pass_temporal'] and not branches and not limits and result['finite'])
    return result,actual,residual,lower,cert,allowance


def run(case,seed_family='independent'):
    oc=verified_oracle_contract()[0];cfg=read(QUAL);g1,collision,natural=model()
    solver=CommonPositionSolver(g1,collision,cfg,natural);g1.assign(natural);bounds=orbit_enclosures(g1)
    oracle=FramewiseReachabilityOracle(g1,collision,oc['oracle'],natural)
    target,hands,ts,source=load_input(case,g1,natural);folder=RUN/case/('dual_position_v1' if seed_family=='independent' else 'dual_position_'+seed_family+'_v1')
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];assert slack<=.002
    dt=float(np.median(np.diff(ts)))
    done=folder/'SOURCE_POSITION_PASS.json'
    if done.exists():print('REUSE_SOURCE_POSITION_PASS',case,flush=True);return
    initial_path=RUN/case/'witness_continuation/independent_0.03.npz'
    if seed_family!='independent':
        ip=folder/'INITIAL_ANCHORS.npz'
        if ip.exists():q0=np.load(ip)['q'].copy()
        else:
            cache=RUN/case/'witness_continuation/INDEPENDENT_NATURAL.npz'
            fresh=np.load(cache)['q'].copy() if cache.exists() else independent_anchors(oracle,target,hands,natural)
            q0=fresh.copy()
            bp=BASELINE/f'{case}.npz'
            if seed_family=='feasible_history' and bp.exists():
                old=np.load(bp)['q'].copy()
                for f in range(len(q0)):
                    olderr=np.linalg.norm(solver.pose_jacobian(old[f],hands[f])[0]-target[f],axis=1)
                    newerr=np.linalg.norm(solver.pose_jacobian(fresh[f],hands[f])[0]-target[f],axis=1)
                    lb=orbit_lower_bounds(target[f],bounds)
                    for arm in range(2):
                        allowed=.01 if lb[arm]<=.01000001 else lb[arm]+slack
                        if olderr[arm]<=allowed or olderr[arm]<newerr[arm]:q0[f,arm*7:(arm+1)*7]=old[f,arm*7:(arm+1)*7]
            # Terminal wrist hinges do not change the position target. Avoid
            # wasting temporal budget on arbitrary independent witness angles.
            q0[:,[6,13]]=natural[[6,13]]
            atomic_npz(ip,q=q0,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=ts)
    elif initial_path.exists():q0=np.load(initial_path)['q'].copy()
    else:
        ip=folder/'INDEPENDENT_SMOOTHED.npz'
        if ip.exists():q0=np.load(ip)['q'].copy()
        else:
            q=independent_anchors(oracle,target,hands,natural)
            q0,fit=refine_trajectory(solver,target,hands,q,.03,100)
            atomic_npz(ip,q=q0,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=ts)
    metrics,actual,residual,lower,cert,allowance=qualify_numeric(solver,q0,target,hands,ts,bounds,slack)
    q=q0.copy();attempts=[]
    for attempt in range(3):
        qp=folder/f'RESTORED_{attempt}.npz';mp=folder/f'RESTORED_{attempt}.json'
        if qp.exists():q=np.load(qp)['q'].copy();fit=read(mp)['fit']
        else:
            print('DUAL_RESTORE',case,attempt,flush=True);start=time.monotonic()
            q,fit=restore(solver,target,hands,q,dt,allowance,max_nfev=240);fit['runtime_s']=time.monotonic()-start
            atomic_npz(qp,q=q,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=ts)
        metrics,actual,residual,lower,cert,allowance=qualify_numeric(solver,q,target,hands,ts,bounds,slack)
        atomic_json(mp,dict(metrics=metrics,fit=fit,source=source,trajectory=file_record(qp)))
        print('DUAL_NUMERIC',case,attempt,'failures',len(metrics['failed_frames']),metrics['temporal'],flush=True)
        attempts.append(file_record(mp))
        if metrics['pass_numeric']:break
    if not metrics['pass_numeric']:
        atomic_json(folder/'NUMERICAL_BLOCKER.json',dict(metrics=metrics,attempts=attempts,next='bounded physical anchor family or necessary-condition analysis; no automatic stop'))
        return
    ranking=folder/'NULLSPACE_RANKING.json'
    if ranking.exists():rank=read(ranking)['rows']
    else:
        rank=rank_constant_nullspace(g1,collision,q,hands)
        atomic_json(ranking,dict(rows=rank,trajectory=file_record(qp),implementation=file_record(ROOT/'tools/common_wrist_nullspace_search.py')))
    for k,row in enumerate(rank[:4]):
        cp=folder/f'NULLSPACE_{k}.npz';gp=folder/f'NULLSPACE_{k}.json'
        candidate=q.copy();candidate[:,[6,13]]=row['wrist_joint_values']
        metrics,actual,residual,lower,cert,allowance=qualify_numeric(solver,candidate,target,hands,ts,bounds,slack)
        if gp.exists():geometry=read(gp)['geometry']
        else:
            geometry=[]
            for f,(v,h) in enumerate(zip(candidate,hands)):
                records=collision.inspect(v,*h);geometry.append(dict(frame=f,records=records))
                if f%150==0:print('DUAL_GEOMETRY',case,k,f,flush=True)
        blocked=[r['frame'] for r in geometry if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
        atomic_npz(cp,EXECUTABLE_Q=candidate,q=candidate,RAW_REPRESENTATION_TARGET=target,EXECUTABLE_FK_POSITION=actual,
            RAW_RESIDUAL_MM=1000*residual,POSITION_CORRECTION_MM=1000*residual,CERTIFIED_LOWER_BOUND_MM=1000*lower,
            SOLVER_OPTIMALITY_GAP_MM=1000*(residual-lower),CERTIFIED_UNREACHABLE=cert,
            RAW_REACHABLE=np.max(residual,axis=1)<=.01,common_hand_q=hands,source_timestamp=ts)
        atomic_json(gp,dict(metrics=metrics,geometry=geometry,blocked_frames=blocked,trajectory=file_record(cp),seed=row))
        print('DUAL_PHYSICAL',case,k,'blocked',len(blocked),'numeric',metrics['pass_numeric'],flush=True)
        if not blocked and metrics['pass_numeric']:
            atomic_json(done,dict(case=case,source=source,trajectory=file_record(cp),verification=file_record(gp),metrics=metrics,
                natural_preparation_join='PENDING',source_target_bytes_unchanged=True,
                implementations=[file_record(ROOT/'tools'/p) for p in ('common_constrained_continuation.py','common_wrist_nullspace_search.py','common_hinge_orbit_bounds.py')]))
            return
    atomic_json(folder/'GEOMETRY_BLOCKER.json',dict(attempts=attempts,next='common collision-aware propagated repair; no waiver'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--seed-family',choices=['independent','feasible_history','unsmoothed_anchors'],default='independent');a=p.parse_args();run(a.case,a.seed_family)
