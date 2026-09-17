#!/usr/bin/env python3
from pathlib import Path
import sys,time
import numpy as np
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_master_autonomous import MASTER,RUN,read,file_record,atomic_json,atomic_text
from tools.cartesian_reachability_forensic import model,BASELINE,DEST
from tools.master_autonomous_provenance import verified_oracle_contract
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.common_grouped_workspace_bounds import grouped_enclosures
from tools.common_g1_position_bounds import residual_lower_bounds


def run():
    oc=verified_oracle_contract()[0];g1,collision,natural=model();g1.assign(natural)
    balls=grouped_enclosures(g1)
    maximum=-np.inf
    for u in qmc.Halton(14,scramble=False).random(2000):
        q=g1.arm_limits[:,0]+u*(g1.arm_limits[:,1]-g1.arm_limits[:,0]);g1.assign(q)
        for s,arm in zip(('left','right'),balls):
            b=arm['balls'][-1];maximum=max(maximum,np.linalg.norm(g1.data.xpos[g1.wrist_ids[s]]-b['center_m'])-b['radius_m'])
    assert maximum<=1e-10
    folder=RUN/'unresolved_audit';atomic_json(folder/'GROUPED_BOUND_PROOF.json',dict(enclosures=balls,halton_validation_count=2000,
        maximum_FK_bound_violation_m=maximum,implementation=file_record(ROOT/'tools/common_grouped_workspace_bounds.py'),
        proof='For unit axis a, ||u+R_a(theta)v||^2 <= (a.u+a.v)^2+(||u_perp||+||v_perp||)^2. Rotations preserve norm. Apply triangle inequality between disjoint translation groups.'))
    pending=[]
    # Discover unresolved records from evidence, not hard-coded outcome labels.
    for p in sorted((DEST/'dense').glob('*_F*.json')):
        d=read(p)
        if d['classification']=='FRAME_REACHABLE' or d['position_infeasibility_certified']:continue
        pending.append((p,d))
    config=dict(oc['oracle']);config['seed_count']=512
    oracle=FramewiseReachabilityOracle(g1,collision,config,natural);rows=[]
    atomic_json(folder/'ADDITIONAL_SEARCH_CONTRACT.json',dict(seed_count=512,maximum_evaluations_per_seed=config['maximum_evaluations_per_seed'],
        selected_frames=[file_record(p) for p,d in pending],selection='all previous bounded-no-witness frames lacking a certificate',
        targets_unchanged=True,collision_rule_unchanged=True,geometry_tolerance_m=1e-5))
    for p,d in pending:
        name=p.stem;out=folder/f'{name}.json'
        if out.exists():rows.append(read(out));continue
        t=np.array(d['incoming_target_m']);h=np.array(d['common_hand_q']);lb=residual_lower_bounds(t,balls)
        cert=bool(lb.max()>.01000001)
        if cert:additional=None;classification='GEOMETRY_CERTIFIED_UNREACHABLE'
        else:
            print('ADDITIONAL_COMMON_SEARCH',name,flush=True)
            additional=oracle.solve(t,h)
            classification='FRAME_REACHABLE' if additional['classification']=='FRAME_REACHABLE' else 'BOUNDED_SEARCH_NO_WITNESS_NOT_CERTIFIED'
        row=dict(source=file_record(p),case=d['case'],frame=d['frame'],classification=classification,
            certified_lower_bound_m=lb.tolist(),additional_search=additional,raw_target=t.tolist(),targets_modified=False)
        atomic_json(out,row);rows.append(row);print('UNRESOLVED_AUDIT',name,classification,lb.max()*1000,flush=True)
    atomic_json(folder/'SUMMARY.json',dict(rows=rows,geometry_bound_proof=file_record(folder/'GROUPED_BOUND_PROOF.json'),
        targets_modified=False,common_method_blind=True))


if __name__=='__main__':run()
