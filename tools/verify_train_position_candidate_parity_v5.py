#!/usr/bin/env python3
"""Verify accepted candidate hashes/raw inputs/timestamps and physical gates."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def main():
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    natural=np.array(read(INITIAL)['g1_14_arm_initial_q_rad']);slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];rows=[]
    matched=[]
    ids=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')['selection']['qualification_ids']
    for ep in ids:
        ta,ha,tsa,sa=load_input(f'WRIST_EP{ep:03d}',g,n);tb,hb,tsb,sb=load_input(f'INTERACTION_EP{ep:03d}',g,n)
        np.testing.assert_array_equal(ha,hb);np.testing.assert_array_equal(tsa,tsb);assert ta.shape==tb.shape
        matched.append(dict(episode=ep,frames=len(tsa),source_clock_exact_equal=True,hand_state_exact_equal=True,
            raw_spatial_targets_may_differ=True,source_A=sa,source_B=sb))
    atomic_json(ST5/'MATCHED_TRAIN11_SOURCE_CLOCK_HAND_PARITY.json',dict(status='PASS',rows=matched,
        note='This is reference clock/hand/input parity, not the unbuilt corrected ACT dataset parity gate'))
    for p in sorted(TRAIN.glob('*/SOURCE_POSITION_PASS.json')):
        r=read(p);case=p.parent.name;path=Path(r['trajectory']['path'])
        assert file_record(path)['sha256']==r['trajectory']['sha256'];assert file_record(Path(r['geometry']['path']))['sha256']==r['geometry']['sha256']
        z=np.load(path);q=z['q'];t,h,ts,source=load_input(case,g,n)
        checks={}
        for key,expected in [('RAW_REPRESENTATION_TARGET',t),('common_hand_q',h),('source_timestamp',ts)]:
            if key in z:
                np.testing.assert_array_equal(z[key],expected);checks[key]='EXACT_ARRAY_EQUAL'
            else:checks[key]='REFERENCED_SOURCE_HASH_AND_INDEPENDENT_QUALIFICATION; field not embedded in legacy smoke candidate'
        m,*_=qualify_numeric(s,q,t,h,ts,bounds,slack);assert m['pass_numeric'],(case,m)
        prefix=preparation_path(natural,q[0],21);np.testing.assert_array_equal(prefix[0],natural)
        dt=float(np.median(np.diff(ts)));tm=temporal_metrics(np.vstack((prefix[:-1],q)),dt,s.config);assert tm['pass_temporal']
        rows.append(dict(case=case,qualification=file_record(p),trajectory=file_record(path),geometry=r['geometry'],
            raw_state_and_timing_checks=checks,source=source,source_numeric=m,complete_temporal=tm,natural_q0_exact=True,preparation_seconds=.7,
            geometry_reuse='Exact trajectory and geometry hashes verified; previously independent all-frame classifier not rerun'))
    out=ST5/'POSITION_CANDIDATE_PARITY_SNAPSHOT.json';atomic_json(out,dict(status='PASS_FOR_ACCEPTED_CANDIDATES_ONLY',rows=rows,
        qualified_count=len(rows),full_train11_qualified=len(rows)==22,common_execution_frozen=False,
        no_raw_target_registration_event_or_gate_change=True,geometry_rule=file_record(ROOT/'tools/common_geometry_confirmed_collision.py'),
        matched_clock_hand_parity=file_record(ST5/'MATCHED_TRAIN11_SOURCE_CLOCK_HAND_PARITY.json'),
        physical_acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json')))
    print('ACCEPTED_CANDIDATE_PARITY',len(rows),'PASS; not a complete common execution freeze',flush=True)

if __name__=='__main__':main()
