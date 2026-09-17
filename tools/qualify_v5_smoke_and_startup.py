#!/usr/bin/env python3
"""Independent common full-source and fixed preparation requalification."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_physical_position_v4 import temporal_metrics
from tools.master_autonomous_common import preparation_path
from tools.common_full_chain_certificate_v5 import full_chain_enclosures

def main():
    verified_oracle_contract();g,c,n=model();s=CommonPositionSolver(g,c,read(QUAL),n)
    assert read(ST5/'full_chain_certificate/VALIDATION.json')['status']=='MODEL_ONLY_OUTER_CERTIFICATE_VALIDATED'
    g.assign(n);bounds=full_chain_enclosures(g);cfg=read(QUAL)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    prep=read(MASTER/'STARTUP_SOURCE_JOIN_CALIBRATION_V2.json');duration=prep['PREP_DURATION_SECONDS'];count=prep['PREP_NUM_FRAMES']
    assert duration==.7 and count==21,'Current locked preparation must not be recalibrated'
    natural=np.array(read(INITIAL)['g1_14_arm_initial_q_rad'])
    opened=np.load(RUN/'startup/WRIST_EP000_COMMON_PREFIX.npz')['common_hand_q'][0]
    folder=V5/'independent_smoke_requalification';rows=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in (0,24,49):
            case=f'{mode}_EP{ep:03d}';p=V5/case/'SOURCE_POSITION_PASS.json'
            if not p.exists():raise RuntimeError(f'No qualified source candidate: {case}')
            record=read(p);qp=Path(record['trajectory']['path']);assert file_record(qp)==record['trajectory']
            z=np.load(qp);q=z['q'].copy();t,h,ts,source=load_input(case,g,n)
            np.testing.assert_array_equal(z['RAW_REPRESENTATION_TARGET'],t)
            np.testing.assert_array_equal(z['source_timestamp'],ts);np.testing.assert_array_equal(z['common_hand_q'],h)
            met,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            prefix=preparation_path(natural,q[0],count);np.testing.assert_array_equal(prefix[0],natural)
            np.testing.assert_allclose(h[0],opened,atol=1e-14,rtol=0)
            full=np.vstack((prefix[:-1],q));hands=np.concatenate((np.repeat(opened[None],count,axis=0),h))
            times=np.r_[np.arange(count)/30,duration+ts-ts[0]]
            temporal=temporal_metrics(full,float(np.median(np.diff(ts))),cfg)
            limits=int(np.count_nonzero((full<g.arm_limits[:,0])|(full>g.arm_limits[:,1])))
            geometry=[dict(global_frame=f,source_frame=f-count,records=c.inspect(v,*hh)) for f,(v,hh) in enumerate(zip(full,hands))]
            hard=[r['global_frame'] for r in geometry if any(x['classification']=='HARD_SELF_COLLISION' for x in r['records'])]
            unresolved=[r['global_frame'] for r in geometry if any(x['classification']=='UNRESOLVED_GEOMETRY' for x in r['records'])]
            proxy=[r['global_frame'] for r in geometry if any(x['classification']=='PROXY_ONLY_OVERLAP' for x in r['records'])]
            passed=bool(met['pass_numeric'] and temporal['pass_temporal'] and not limits and not hard and not unresolved)
            gp=folder/f'{case}_GEOMETRY.json';atomic_json(gp,dict(geometry=geometry))
            out=folder/f'{case}_PREPARED.npz'
            atomic_npz(out,q=full,EXECUTABLE_Q=full,common_hand_q=hands,execution_timestamp=times,source_timestamp=ts,
                RAW_REPRESENTATION_TARGET=t,raw_tracking_mask=np.r_[np.zeros(count,bool),np.ones(len(q),bool)],
                source_frame_index=np.r_[np.full(count,-1),np.arange(len(q))],PREP_DURATION_SECONDS=np.array(duration),
                POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb)
            row=dict(case=case,qualified=passed,source_metrics=met,complete_temporal=temporal,hard_limit_violations=limits,
                hard_collision_frames=hard,unresolved_geometry_frames=unresolved,proxy_only_overlap_frames=proxy,
                natural_q0_exact=True,preparation_seconds=duration,preparation_frames=count,source=source,
                trajectory=file_record(out),source_trajectory=file_record(qp),geometry=file_record(gp))
            rows.append(row);atomic_json(folder/f'{case}_QUALIFICATION.json',row)
            print('INDEPENDENT_POSITION',case,passed,'hard',len(hard),'unresolved',len(unresolved),flush=True)
    result=dict(status='SMOKE3_COMMON_EXECUTABLE_POSITION_QUALIFIED' if all(r['qualified'] for r in rows) else 'POSITION_RECOVERY_REQUIRED',
        rows=rows,qualified_counts={mode:sum(r['qualified'] for r in rows if r['case'].startswith(mode)) for mode in ('WRIST','INTERACTION')},
        training11_complete=False,full6d_not_run=True,simulation_contact_not_tested=True,
        next='FIXED_TRAIN11_POSITION_QUALIFICATION',contract=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'))
    rp=ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json';atomic_json(rp,result)
    log('SMOKE3_POSITION_AND_FIXED_PREPARATION',result['status'],[rp], 'Independent requalification after local geometry repair',
        'COMMON_REDUNDANCY_RECOVERY', 'Rechecked all source states, exact raw targets, fixed 0.7s preparation, per-joint limits, adaptive branches and detailed geometry',
        [rp],result['next'],f"공통 최종 검증: A {result['qualified_counts']['WRIST']}/3, B {result['qualified_counts']['INTERACTION']}/3. 자연 q0와 0.7초 준비 구간을 유지했습니다. 고정 TRAIN11 검증 및 후속 단계를 계속합니다.")
    if not all(r['qualified'] for r in rows):raise RuntimeError('Common source/preparation qualification needs recovery')

if __name__=='__main__':main()
