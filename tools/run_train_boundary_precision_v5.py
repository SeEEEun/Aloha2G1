#!/usr/bin/env python3
"""Bounded common precision continuation retaining discovered geometry planes.

Search planes are not acceptance certificates. Full original classification,
raw-position and physical temporal checks remain mandatory after continuation.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.common_geometry_boundary_v5 import GeometryBoundarySolver
from tools.common_conic_trust_realization_v6 import solve
from tools.qualify_train_candidate_geometry_v5 import run as qualify

def run(case,boundary_folder):
    verified_oracle_contract();g,c,n=model();base=Path(boundary_folder).resolve();paths=sorted(base.glob('CANDIDATE_*.npz'))
    assert paths;src=paths[-1];q=np.load(src)['q'];t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)))
    locals={};goals={};inputs=[]
    for p in sorted(base.glob('BOUNDARY_*.json')):
        if 'UNBRACKETED' in p.name:continue
        parts=p.stem.split('_');f=int(parts[2]);pair=(int(parts[3]),int(parts[4]));boundary=read(p)
        if 'point' not in boundary:continue
        local=locals.setdefault(f,GeometryBoundarySolver(g,c,read(QUAL),n));local.boundaries[pair]=boundary
        goals.setdefault(f,[])
        if (*pair,f) not in goals[f]:goals[f].append((*pair,f))
        inputs.append(file_record(p))
    class Dispatch(GeometryBoundarySolver):
        def clearance_values(self,pairs):
            local=locals[int(pairs[0][2])];local.current_q=self.current_q
            return local.clearance_values([p[:2] for p in pairs])
    s=Dispatch(g,c,read(QUAL),n);g.assign(n);bounds=full_chain_enclosures(g)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    folder=base/'common_boundary_precision_v1';atomic_json(folder/'CONTRACT.json',dict(input=file_record(src),source=source,
        geometry_planes=inputs,maximum_iterations_per_arm=500,window_padding=24,trust_radius=.15,raw_targets_or_gates_changed=False))
    m,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
    frames=sorted(set(goals)|set(m['failed_frames']))
    if not frames:raise ValueError('No precision-recovery interval')
    begin=max(0,min(frames)-24);end=min(len(q),max(frames)+25)
    for arm in range(2):
        pp={f-begin:pairs for f,pairs in goals.items() if begin<=f<end}
        q[begin:end],fit=solve(s,t[begin:end],h[begin:end],q[begin:end],allow[begin:end],dt,arm,pp,trust_radius=.15,max_iterations=500)
        atomic_json(folder/f'FIT_ARM_{arm}.json',fit)
    out=folder/'CANDIDATE.npz';atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
    qualify(case,str(out));atomic_json(folder/'BOUNDED_RESULT.json',dict(status='COMMON_BOUNDARY_PRECISION_COMPLETE',
        qualification=file_record(folder/'CANDIDATE_independent_qualification/FINAL_QUALIFICATION.json'),global_infeasibility_proven=False))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
