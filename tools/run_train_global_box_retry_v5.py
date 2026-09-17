#!/usr/bin/env python3
"""Independent bounded box search for unresolved TRAIN oracle samples.

Differential evolution is a search, NOT a mathematical global certificate.
No raw targets, acceptance thresholds, or articulation/model assets are edited.
"""
from pathlib import Path
import sys,time
import numpy as np
from scipy.optimize import differential_evolution
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def run(case,audit_path):
    oc=verified_oracle_contract()[0];g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    oracle=FramewiseReachabilityOracle(g,c,oc['oracle'],n);t,h,ts,source=load_input(case,g,n)
    audit=Path(audit_path).resolve();summary=read(audit/'COMPLETE.json');folder=audit/'common_global_box_retry_v1'
    atomic_json(folder/'CONTRACT.json',dict(input=file_record(audit/'COMPLETE.json'),source=source,
        frames=summary['no_witness'],population_multiplier=16,maximum_generations=400,random_seed=0,
        polish_evaluations=300,geometry_rule_unchanged=True,raw_targets_changed=False,global_certificate=False))
    rows=[]
    for f in summary['no_witness']:
        dest=folder/f'FRAME_{f:04d}.json'
        if dest.exists():rows.append(read(dest));continue
        previous=read(audit/f'FRAME_{f:04d}.json');q=np.array(previous['candidates'][0]['q']);allow=np.array(previous['allowance_m'])
        started=time.monotonic();fits=[]
        for arm in np.flatnonzero(np.array(previous['candidates'][0]['residual_m'])>allow):
            ids=np.arange(7*arm,7*arm+6)
            def objective(v):
                value=q.copy();value[ids]=v
                error=s.pose_jacobian(value,h[f])[0][arm]-t[f,arm]
                return float(error@error)
            fit=differential_evolution(objective,list(zip(s.lower[ids],s.upper[ids])),seed=0,popsize=16,maxiter=400,
                tol=1e-10,atol=1e-16,polish=False,workers=1,updating='immediate')
            q[ids]=fit.x;fits.append(dict(arm=int(arm),evaluations=int(fit.nfev),generations=int(fit.nit),
                best_squared_residual_m2=float(fit.fun),optimizer_success=bool(fit.success)))
        q,fit=oracle.fit(t[f],h[f],q,max_evaluations=300)
        geometry=c.inspect(q,*h[f]);error=np.array(fit['residual_m'])
        valid=bool(np.all(error<=allow)) and not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in geometry)
        result=dict(frame=f,q=q.tolist(),residual_m=error.tolist(),allowance_m=allow.tolist(),geometry=geometry,
            framewise_witness=valid,global_infeasibility_proven=False,box_search=fits,polish=fit,runtime_s=time.monotonic()-started)
        atomic_json(dest,result);rows.append(result);print('COMMON_GLOBAL_BOX_RETRY',case,f,'witness',valid,'residual_mm',error*1000,flush=True)
    atomic_json(folder/'COMPLETE.json',dict(results=rows,global_infeasibility_proven=False,
        remaining_unresolved=[r['frame'] for r in rows if not r['framewise_witness']]))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
