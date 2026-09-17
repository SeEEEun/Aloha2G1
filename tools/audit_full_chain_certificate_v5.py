#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_local_redundancy_v5 import *
from tools.common_full_chain_certificate_v5 import full_chain_enclosures

def main():
    verified_oracle_contract();g,c,n=model();g.assign(n);bounds=full_chain_enclosures(g)
    folder=ST5/'full_chain_certificate';atomic_json(folder/'MODEL_ONLY_CERTIFICATE.json',dict(enclosures=bounds,
        raw_gate_m=.01,numerical_optimality_slack_m=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'],
        acceptance_thresholds_changed=False,implementation=file_record(ROOT/'tools/common_full_chain_certificate_v5.py')))
    print('MODEL_ONLY_RADII',[(r['side'],r['full_chain_certificate']) for r in bounds],flush=True)
    errors=[];maximum=0.
    for u in qmc.Halton(14,scramble=False).random(5000):
        q=g.arm_limits[:,0]+u*(g.arm_limits[:,1]-g.arm_limits[:,0]);g.assign(q)
        pos=np.array([g.data.xpos[g.wrist_ids[s]].copy() for s in ('left','right')]);lb=orbit_lower_bounds(pos,bounds)
        maximum=max(maximum,float(lb.max()))
    rows=[]
    for p in sorted((DEST/'dense').glob('*_F*.json')):
        d=read(p);lb=orbit_lower_bounds(np.array(d['incoming_target_m']),bounds)
        actual=np.array(d['best']['residual_m'])
        valid=bool(np.all(lb<=actual+1e-8))
        if d['classification']=='FRAME_REACHABLE':valid &= bool(lb.max()<=.01000001)
        if not valid:errors.append(str(p))
        rows.append(dict(case=d['case'],frame=d['frame'],lower_bound_mm=(1000*lb).tolist(),witness_residual_mm=(1000*actual).tolist(),consistent=valid))
    result=dict(status='MODEL_ONLY_OUTER_CERTIFICATE_VALIDATED' if not errors and maximum<=1e-10 else 'INVALID_CERTIFICATE_DO_NOT_USE',
        random_FK_checks=5000,maximum_random_FK_lower_bound_m=maximum,dense_saved_witness_checks=len(rows),errors=errors,rows=rows,
        proof_not_replaced_by_sampling=True,thresholds_changed=False,method_identity_inputs=False,raw_target_inputs_to_radius=False,
        source_model_certificate=file_record(folder/'MODEL_ONLY_CERTIFICATE.json'))
    atomic_json(folder/'VALIDATION.json',result)
    print('FULL_CHAIN_CERTIFICATE',result['status'],'errors',errors,flush=True)
    assert result['status']=='MODEL_ONLY_OUTER_CERTIFICATE_VALIDATED'

if __name__=='__main__':main()
