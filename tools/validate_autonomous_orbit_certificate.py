#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_master_autonomous import MASTER,RUN,read,file_record,atomic_json,atomic_text
from tools.cartesian_reachability_forensic import model,BASELINE,DEST
from tools.master_autonomous_provenance import verified_oracle_contract
from tools.common_hinge_orbit_bounds import orbit_enclosures,orbit_lower_bounds
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle


def run():
    oc=verified_oracle_contract()[0];g1,collision,natural=model();g1.assign(natural);bounds=orbit_enclosures(g1)
    maximum=0.
    for u in qmc.Halton(14,scramble=False).random(5000):
        q=g1.arm_limits[:,0]+u*(g1.arm_limits[:,1]-g1.arm_limits[:,0]);g1.assign(q)
        pos=np.array([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
        maximum=max(maximum,float(orbit_lower_bounds(pos,bounds).max()))
    assert maximum<=1e-10
    rows=[];certified=[]
    for p in sorted((DEST/'dense').glob('*_F*.json')):
        d=read(p);lb=orbit_lower_bounds(np.array(d['incoming_target_m']),bounds)
        cert=bool(lb.max()>.01000001)
        if d['best']['geometry_valid']:
            assert lb.max()<=d['best']['maximum_residual_m']+1e-8
        if d['classification']=='FRAME_REACHABLE':assert not cert
        row=dict(case=d['case'],frame=d['frame'],certified_lower_bound_mm=(lb*1000).tolist(),
            certified_unreachable=cert,prior_certified=d['position_infeasibility_certified'],source=file_record(p))
        rows.append(row)
        if cert:certified.append((p,d,lb))
    folder=RUN/'orbit_certificate'
    atomic_json(folder/'GEOMETRIC_CERTIFICATE.json',dict(enclosures=bounds,validation_random_FK_count=5000,
        maximum_FK_lower_bound_m=maximum,all_602_existing_witnesses_consistent=True,rows=rows,
        implementation=file_record(ROOT/'tools/common_hinge_orbit_bounds.py'),
        proof='Distance to circle centerline is sqrt((norm(v_perpendicular)-circle_radius)^2+v_axial^2). Subtract the conservative remaining-chain ball radius. This relaxes all later joint choices and all collision constraints, so it cannot overstate the minimum residual.'))
    # Deterministic repeatability measured only on fixed TRAIN-side spread.
    chosen=np.linspace(0,len(certified)-1,11).round().astype(int)
    oracle=FramewiseReachabilityOracle(g1,collision,oc['oracle'],natural);repeat=[]
    corrections={(d['case'],d['frame']):d for d in read(DEST/'CORRECTION_BOUND_WITNESSES.json')}
    atomic_json(folder/'REPEATABILITY_CONTRACT.json',dict(selected=[file_record(certified[i][0]) for i in chosen],
        seed_policy='cached collision-valid witness and 4 deterministic 0.01rad sinusoidal joint perturbations',
        max_evaluations_per_fit=180,slack_rule='max(existing 10um geometry numerical resolution, maximum observed deterministic-fit residual range); fail if above user 2mm cap',
        no_success_or_method_input=True))
    for i in chosen:
        p,d,lb=certified[i];t=np.array(d['incoming_target_m']);h=np.array(d['common_hand_q'])
        initial=np.array(corrections[(d['case'],d['frame'])]['witness_q']);trials=[]
        for k in range(5):
            seed=initial if k==0 else initial+.01*np.sin(np.arange(14)+k)
            q,info=oracle.fit(t,h,seed,max_evaluations=180)
            rr=collision.inspect(q,*h)
            trials.append(dict(q=q.tolist(),geometry=rr,geometry_valid=not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in rr),**info))
        spread=max(r['maximum_residual_m'] for r in trials)-min(r['maximum_residual_m'] for r in trials)
        repeat.append(dict(case=d['case'],frame=d['frame'],residual_range_m=spread,trials=trials,lower_bound_m=lb.tolist()))
        print('REPEATABILITY',d['case'],d['frame'],spread*1000,flush=True)
        atomic_json(folder/'REPEATABILITY_PROGRESS.json',repeat)
    slack=max(1e-5,max(r['residual_range_m'] for r in repeat))
    result=dict(rows=repeat,numerical_slack_m=slack,user_cap_m=.002,within_cap=slack<=.002,
        geometry_numerical_floor_m=1e-5,certified_count=sum(r['certified_unreachable'] for r in rows),
        newly_certified_count=sum(r['certified_unreachable'] and not r['prior_certified'] for r in rows),
        repeatability_is_not_a_global_optimality_proof=True)
    atomic_json(folder/'REPEATABILITY_RESULT.json',result)
    atomic_text(folder/'GEOMETRIC_CERTIFICATE.md',f'# Common geometry certificate and numerical repeatability\n\nAll 602 old failed frames reclassified from their unchanged stored targets; no dense search rerun. {result["certified_count"]} certified outside the 10mm gate, including {result["newly_certified_count"]} formerly unresolved frames. The first-hinge orbit is retained rather than relaxed into a full sphere. The inequality is analytic; 5000 FK points and every existing valid witness are consistency checks, not the proof. Numerical slack measured from 11 deterministic TRAIN-side cases: {1000*slack:.9f}mm; user cap respected: {slack<=.002}. This is not a trajectory qualification.\n')


if __name__=='__main__':run()
