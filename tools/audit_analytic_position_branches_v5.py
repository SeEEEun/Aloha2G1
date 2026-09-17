#!/usr/bin/env python3
"""Random model-only roundtrip verification; no source target modification."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_analytic_position_branches_v5 import AnalyticPositionBranches

verified_oracle_contract();g,c,n=model();analytic=AnalyticPositionBranches(g);rng=np.random.default_rng(0)
h=np.zeros((2,7));rows=[]
for i in range(200):
    q=rng.uniform(g.arm_limits[:,0]+1e-4,g.arm_limits[:,1]-1e-4);g.assign(q,*h)
    targets=[g.wrist_pose(side)[:3,3].copy() for side in ('left','right')]
    for arm in range(2):
        candidates=analytic.candidates(targets[arm],q[arm*7:arm*7+3],q,h,arm)
        assert candidates,(i,arm,q)
        best=min(candidates,key=lambda r:np.linalg.norm(r['q']-q));error=float(np.linalg.norm(best['q']-q))
        assert error<1e-7,(i,arm,error)
        rows.append(dict(sample=i,arm=arm,branches=len(candidates),q_roundtrip_rad=error,fk_residual_m=best['residual_m']))
folder=ST5/'analytic_branch_audit';atomic_json(folder/'MODEL_ONLY_ROUNDTRIP.json',dict(status='PASS',rows=rows,
    seed=0,source_target_inputs=False,implementation=file_record(ROOT/'tools/common_analytic_position_branches_v5.py')))
print('ANALYTIC_MODEL_ROUNDTRIP',len(rows),'PASS',flush=True)
