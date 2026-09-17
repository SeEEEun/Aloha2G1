#!/usr/bin/env python3
"""Qualify the actual common physical Dex3 nominal commands without arm rescue."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_reference_physics import *
from tools.final_paper_position_run import model,inspect

def qualify_action(row):
    folder=DEST/'complete_action'/row['key'];rp=folder/'RESULT.json'
    if rp.exists():return read(rp)
    six=read(DEST/'full6d'/row['key']/'RESULT.json')
    if six['outcome']!='FULL6D_EXECUTABLE':
        r=dict(outcome='NOT_RUN_FULL6D_FAILURE',case=row,full6d_outcome=six['outcome']);atomic_json(rp,r);return r
    cp,_=command(row,six['trajectory']['path'])
    with np.load(cp) as z:q=z['commanded_q_rad'];names=z['joint_names'].astype(str)
    g,c,n=model();expected=[*g.arm_joint_names,*g.hand_joint_names['left'],*g.hand_joint_names['right']]
    assert len(names)==len(expected)==28 and set(names)==set(expected)
    # The authoritative PhysX vector orders middle before index; the geometry
    # model orders index before middle. Match by name, never by assumed index.
    permutation=np.array([list(names).index(name) for name in expected])
    model_q=q[:,permutation]
    rows=inspect(c,model_q[:,:14],model_q[:,14:].reshape(-1,2,7))
    counts={k:sum(any(v['classification']==k for v in row['records']) for row in rows) for k in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY','PROXY_ONLY_OVERLAP')}
    lo,hi,_=authoritative_joint_limits(read(JOINT));violations=int(((q<lo)|(q>hi)).sum())
    valid=not counts['HARD_SELF_COLLISION'] and not counts['UNRESOLVED_GEOMETRY'] and not violations and np.isfinite(q).all()
    outcome='COMPLETE_ACTION_EXECUTABLE' if valid else ('COMPLETE_ACTION_HARD_COLLISION' if counts['HARD_SELF_COLLISION'] else 'COMPLETE_ACTION_UNRESOLVED_OR_LIMIT_INVALID')
    atomic_json(folder/'GEOMETRY.json',dict(rows=rows))
    r=dict(outcome=outcome,case=row,command=file_record(cp),geometry_counts=counts,commanded_limit_violations=violations,geometry=file_record(folder/'GEOMETRY.json'),
        explanation='Position/6D diagnostics used common reference hand states. The final physical P14 realization is also checked, with the unchanged detailed classifier. No arm or target repair is permitted.',full6d_source=six['trajectory'],command_to_geometry_permutation=permutation.tolist(),named_joint_values_preserved=True)
    atomic_json(rp,r);print('COMPLETE_ACTION',row['key'],outcome,flush=True);return r

if __name__=='__main__':
    rows=cases();done=set()
    while len(done)<len(rows):
        for row in rows:
            if row['key'] not in done and (DEST/'full6d'/row['key']/'RESULT.json').exists():
                six=read(DEST/'full6d'/row['key']/'RESULT.json')
                if six['outcome']=='INFRASTRUCTURE_INVALID':continue
                qualify_action(row);done.add(row['key'])
        if len(done)<len(rows):time.sleep(15)
