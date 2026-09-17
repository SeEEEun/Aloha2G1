#!/usr/bin/env python3
"""Non-promoting common diagnostic of internal trust cap vs physical gates.

Neither the frozen config nor the authoritative qualification is changed.
The aggregate cap is replaced ONLY IN THIS DIAGNOSTIC SEARCH by the redundant
bound implied mathematically by the unchanged 14 per-joint velocity bounds.
Results retain both strict and original-evaluator interpretations.
"""
from pathlib import Path
import sys, argparse, copy, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_fixed_anchor_precision import restore


def run(case):
    verified_oracle_contract();g,c,natural=model();s=CommonPositionSolver(g,c,read(QUAL),natural)
    g.assign(natural);bounds=orbit_enclosures(g)
    t,h,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)))
    folder=RUN/case/'aggregate_step_semantics_diagnostic_v1'
    init=RUN/case/'local_dense_continuation_v1/PASS_2_FULL.npz'
    ap=RUN/case/'certified_anchor_recovery/CERTIFIED_ANCHORS.npz'
    a=np.load(ap);anchors=a['q'];mask=a['fixed_mask'];q=np.load(init)['q'].copy()
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    diagnostic=copy.copy(s);diagnostic.config=dict(s.config)
    diagnostic.config['maximum_step_norm_rad']=float(np.sqrt(14)*min(s.config['maximum_joint_step_rad'],s.config['maximum_velocity_rad_s']*dt))
    atomic_json(folder/'CONTRACT.json',dict(status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',source=file_record(init),anchors=file_record(ap),
        physical_gates_unchanged=True,frozen_configuration_unchanged=True,
        diagnostic_aggregate_bound_rad=diagnostic.config['maximum_step_norm_rad'],
        derivation='sqrt(14) times unchanged per-joint step bound; mathematically redundant, not selected from an A/B outcome',
        maximum_attempts=2,maximum_evaluations_per_attempt=400,
        implementations=[file_record(Path(__file__)),file_record(ROOT/'tools/common_fixed_anchor_precision.py')]))
    for attempt in range(2):
        p=folder/f'ATTEMPT_{attempt}.npz';mp=folder/f'ATTEMPT_{attempt}.json'
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        if p.exists():q=np.load(p)['q'].copy();fit=read(mp)['fit']
        else:
            start=time.monotonic();print('STEP_SEMANTICS_START',case,attempt,flush=True)
            q,fit=restore(diagnostic,t,h,q,dt,allow,max_nfev=400,fixed_mask=mask,fixed_values=anchors)
            fit['runtime_s']=time.monotonic()-start
            atomic_npz(p,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
        tm=met['temporal'];cfg=s.config
        original_temporal=bool(tm['maximum_velocity_rad_s']<=cfg['maximum_velocity_rad_s']+1e-7 and tm['maximum_acceleration_rad_s2']<=cfg['maximum_acceleration_rad_s2']+1e-5 and tm['maximum_scalar_step_rad']<=cfg['maximum_joint_step_rad']+1e-7)
        original_numeric=bool(original_temporal and not met['failed_frames'] and not met['branch_discontinuities'] and not met['hard_limit_violations'] and met['finite'])
        atomic_json(mp,dict(status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',fit=fit,strict_metrics=met,
            original_evaluator_temporal_pass=original_temporal,original_evaluator_numeric_pass=original_numeric,
            detailed_geometry='NOT_YET_CHECKED',trajectory=file_record(p)))
        print('STEP_SEMANTICS_RESULT',case,attempt,'failures',met['failed_frames'],'original_numeric',original_numeric,'strict',met['pass_numeric'],tm,flush=True)
        if original_numeric:break
    atomic_json(folder/'COMPLETE.json',dict(status='DIAGNOSTIC_ONLY_NOT_QUALIFICATION',last_result=file_record(mp),global_impossibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');run(p.parse_args().case)
