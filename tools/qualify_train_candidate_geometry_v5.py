#!/usr/bin/env python3
"""Shared full numeric, detailed geometry and fixed-startup qualification."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *

def run(case,path):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    g.assign(n);bounds=full_chain_enclosures(g);t,h,ts,source=load_input(case,g,n)
    path=Path(path).resolve();q=np.load(path)['q'];dt=float(np.median(np.diff(ts)))
    folder=path.parent/(path.stem+'_independent_qualification');record=file_record(path)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    m,act,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
    atomic_json(folder/'NUMERIC.json',dict(metrics=m,trajectory=record))
    geometry=[]
    for f,(v,hh) in enumerate(zip(q,h)):
        p=folder/'frames'/f'{f:04d}.json'
        if p.exists():r=read(p)
        else:r=dict(frame=f,records=c.inspect(v,*hh));atomic_json(p,r)
        geometry.append(r)
        if f%100==0:print('TRAIN_GEOMETRY',case,f,flush=True)
    gp=folder/'COMPLETE_GEOMETRY.json';atomic_json(gp,dict(geometry=geometry,trajectory=record))
    hard=[r['frame'] for r in geometry if any(x['classification']=='HARD_SELF_COLLISION' for x in r['records'])]
    unresolved=[r['frame'] for r in geometry if any(x['classification']=='UNRESOLVED_GEOMETRY' for x in r['records'])]
    prefix=preparation_path(np.array(read(INITIAL)['g1_14_arm_initial_q_rad']),q[0],21)
    tm=temporal_metrics(np.vstack((prefix[:-1],q)),dt,s.config)
    pg=[dict(frame=f,records=c.inspect(v,*h[0])) for f,v in enumerate(prefix[:-1])]
    pb=[r['frame'] for r in pg if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
    out=folder/'QUALIFIED_FIELDS.npz';atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts,
        EXECUTABLE_FK_POSITION=act,POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),CERTIFIED_UNREACHABLE=cert)
    result=dict(case=case,metrics=m,trajectory=file_record(out),input_trajectory=record,geometry=file_record(gp),
        hard_collision_frames=hard,unresolved_geometry_frames=unresolved,blocked_frames=sorted(set(hard+unresolved)),
        preparation_blocked_frames=pb,preparation_geometry=pg,complete_temporal=tm,source=source,
        qualified=m['pass_numeric'] and not hard and not unresolved and not pb and tm['pass_temporal'])
    atomic_json(folder/'FINAL_QUALIFICATION.json',result)
    if result['qualified'] and not (TRAIN/case/'SOURCE_POSITION_PASS.json').exists():atomic_json(TRAIN/case/'SOURCE_POSITION_PASS.json',result)
    print('TRAIN_INDEPENDENT',case,result['qualified'],'hard',hard,'unresolved',unresolved,'prep',pb,flush=True)

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
