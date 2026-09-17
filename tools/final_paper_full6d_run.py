#!/usr/bin/env python3
"""Common position-anchored invocation of the repository's existing 6D DLS.

Raw orientation is never calibrated twice or rewritten. A bounded failure is
an outcome, not permission to tune this stage or to certify impossibility.
"""
from pathlib import Path
import argparse,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_run import *
from tools.doll_handoff_retargeting.retarget import SharedTemporalIK
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
F6=OUT/'03_common_execution_freeze/COMMON_FULL6D_FREEZE_MANIFEST.json'

def freeze6():
    verify()
    if F6.exists():return verify6()
    common=load_common_config(RESET/'config/common_config.json')
    contract=dict(created_at=now(),position_freeze=file_record(FREEZE),
        configuration=common['shared_temporal_ik'],natural_posture=common['natural_arm_redundancy'],
        orientation_chain='Already registered active right-handed model wrist rotations. No additional TCP calibration. SciPy rotation matrices/XYZW; MuJoCo xmat matrices.',
        orientation_audit=file_record(ST5/'orientation_interface_audit/ORIENTATION_CHAIN_AUDIT.json'),
        source_position='Raw target and certified allowances unchanged. Qualified executable FK is the 6D positional objective; whole final trajectory must still satisfy the original raw/closest-feasible position contract.',
        seeds='At each frame, qualified position q and previous 6D q; deterministic selection on Cartesian then angular residual, geometry, continuity. No label/episode input.',
        budget='Existing 250 initial/40 subsequent iterations per seed; two seeds; existing 9-frame order-3 smoothing then 25-iteration reprojection. No outcome-dependent recovery or expanded budget.',
        acceptance='Every source frame: previous position acceptance, raw angular residual <=0.75 rad, finite, limits, common physical temporal and detailed geometry acceptance including unchanged0.7s prefix.',
        angular_projection='Not automatically invoked without an orientation-infeasibility certificate. Residual is reported, bounded failure does not prove impossibility.',
        records=[file_record(Path(__file__)),file_record(ROOT/'tools/doll_handoff_retargeting/retarget.py'),file_record(RESET/'config/common_config.json')])
    atomic_json(F6,contract);atomic_json(DEST/'FULL6D_FREEZE_SHA256.json',file_record(F6))
    return contract

def verify6():
    contract=read(F6);assert file_record(F6)==read(DEST/'FULL6D_FREEZE_SHA256.json')
    for r in contract['records']:assert file_record(Path(r['path']))==r
    return contract

def orient(g,q,rot):
    executed=[]
    for v in q:
        g.assign(v);executed.append([g.data.xmat[g.wrist_ids[side]].reshape(3,3).copy() for side in ('left','right')])
    executed=np.array(executed)
    err=Rotation.from_matrix((rot@executed.transpose(0,1,3,2)).reshape(-1,3,3)).magnitude().reshape(len(q),2)
    return executed,err

def solve6(common,g,n,t,rot,q0,folder):
    """No method or episode inputs. Existing DLS numeric core is unmodified."""
    solver=SharedTemporalIK(common,g,n);cfg=solver.config
    targets={f'{side}_wrist_position':t[:,k] for k,side in enumerate(('left','right'))}
    targets.update({f'{side}_wrist_rotation':rot[:,k] for k,side in enumerate(('left','right'))})
    p=folder/'FORWARD.npz'
    if p.exists():raw=np.load(p)['q']
    else:
        raw=np.empty_like(q0);previous=q0[0].copy();previous2=previous.copy();reports=[]
        for f in range(len(q0)):
            iterations=cfg['max_iterations_initial_frame'] if f==0 else cfg['max_iterations_per_frame']
            candidates=[solver._solve_seed(targets,f,seed,previous,previous2,iterations,None if f==0 else cfg['max_frame_joint_step_rad']) for seed in (q0[f],previous)]
            value,meta=min(candidates,key=lambda x:(not x[1]['accepted'],x[1]['position_error_max_m']/cfg['position_tolerance_m']+x[1]['orientation_error_max_rad']/cfg['orientation_tolerance_rad']))
            raw[f]=value;reports.append(meta);previous2,previous=previous,value.copy()
            if f%100==0:print('COMMON_FULL6D_FORWARD',f,flush=True)
        atomic_npz(p,q=raw);atomic_json(folder/'FORWARD.json',reports)
    width=min(cfg['temporal_smoothing_window'],len(raw) if len(raw)%2 else len(raw)-1)
    smooth=np.clip(savgol_filter(raw,width,cfg['temporal_smoothing_polyorder'],axis=0,mode='interp'),g.arm_limits[:,0],g.arm_limits[:,1])
    final=np.empty_like(raw);previous=smooth[0].copy();previous2=previous.copy();reports=[]
    for f in range(len(raw)):
        value,meta=solver._solve_seed(targets,f,smooth[f],previous,previous2,cfg['max_iterations_reprojection'],None if f==0 else cfg['reprojection_max_frame_joint_step_rad'])
        final[f]=value;reports.append(meta);previous2,previous=previous,value.copy()
    atomic_json(folder/'REPROJECTION.json',reports)
    return final

def run6(key):
    pc=verify();contract=verify6();row=next(x for x in pc['cases'] if x['key']==key)
    prior=read(DEST/'position'/key/'RESULT.json');folder=DEST/'full6d'/key;rp=folder/'RESULT.json'
    if rp.exists() and read(rp)['outcome']!='INFRASTRUCTURE_INVALID':return
    if prior['outcome']!='POSITION_EXECUTABLE':
        atomic_json(rp,dict(outcome='NOT_RUN_POSITION_FAILURE',position_outcome=prior['outcome'],case=row,freeze=file_record(F6)));return
    try:
        start=time.monotonic();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
        t,rot,h,ts,opened=inputs(row,g,n);dt=float(np.median(np.diff(ts)));z=np.load(prior['selected']['trajectory']['path'])
        q0=z['q'];execpos=z['EXECUTABLE_FK_POSITION'];common=load_common_config(RESET/'config/common_config.json')
        candidate=folder/'CANDIDATE_Q.npz'
        if candidate.exists():q=np.load(candidate)['q']
        else:q=solve6(common,g,n,execpos,rot,q0,folder);atomic_npz(candidate,q=q)
        met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,pc['acceptance']['certified_slack_m'])
        rotation,error=orient(g,q,rot);prefix=preparation_path(np.array(pc['acceptance']['natural_q0']),q[0],21);full=np.vstack((prefix[:-1],q))
        temporal=temporal_metrics(full,dt,s.config);gp=folder/'GEOMETRY.json'
        if gp.exists():geometry=read(gp)
        else:
            geometry=dict(source=inspect(c,q,h),preparation=inspect(c,prefix[:-1],np.repeat(opened[None],21,axis=0)));atomic_json(gp,geometry)
        rows=geometry['source']+geometry['preparation'];counts={name:sum(any(x['classification']==name for x in r['records']) for r in rows) for name in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY','PROXY_ONLY_OVERLAP')}
        if not met['finite'] or not np.isfinite(error).all() or met['hard_limit_violations']:outcome='FULL6D_NUMERICAL_INVALID'
        elif counts['HARD_SELF_COLLISION']:outcome='FULL6D_HARD_COLLISION'
        elif not temporal['pass_temporal']:outcome='FULL6D_TEMPORAL_INVALID'
        elif met['failed_frames'] or counts['UNRESOLVED_GEOMETRY'] or np.any(error>common['shared_temporal_ik']['orientation_tolerance_rad']):outcome='FULL6D_NO_SOLUTION_WITHIN_COMMON_BUDGET'
        else:outcome='FULL6D_EXECUTABLE'
        path=folder/'TRAJECTORY.npz'
        atomic_npz(path,q=q,full_q=full,full_hand_q=np.concatenate((np.repeat(opened[None],21,axis=0),h)),
            common_hand_q=h,source_timestamp=ts,execution_timestamp=np.r_[np.arange(21)/30,ts-ts[0]+.7],
            RAW_REPRESENTATION_TARGET=t,RAW_ORIENTATION_TARGET=rot,EXECUTABLE_FK_POSITION=act,EXECUTABLE_ORIENTATION=rotation,
            ORIENTATION_RESIDUAL_RAD=error,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb)
        atomic_json(rp,dict(outcome=outcome,case=row,freeze=file_record(F6),position_input=prior['selected']['trajectory'],trajectory=file_record(path),
            metrics=met,temporal=temporal,geometry_counts=counts,geometry=file_record(gp),
            orientation_residual_rad=dict(mean=float(error.mean()),p95=float(np.quantile(error,.95)),max=float(error.max())),
            orientation_failed_frames=np.flatnonzero(error.max(axis=1)>.75).tolist(),runtime_s=time.monotonic()-start,
            position_objective_displacement_mm=float(np.linalg.norm(act-execpos,axis=2).max()*1000),raw_targets_changed=False))
        print('FULL6D_FINAL',key,outcome,flush=True)
    except Exception:
        atomic_json(rp,dict(outcome='INFRASTRUCTURE_INVALID',case=row,freeze=file_record(F6),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args()
    freeze6() if a.action=='freeze' else run6(a.action)
