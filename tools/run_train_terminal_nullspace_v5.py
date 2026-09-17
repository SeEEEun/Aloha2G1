#!/usr/bin/env python3
"""Common constant position-null wrist seed ranking with full geometry audits."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.qualify_train_candidate_geometry_v5 import run as qualify

def run(case,path):
    verified_oracle_contract();g,c,n=model();s=RobustProxyPenaltySolver(g,c,read(QUAL),n)
    t,h,ts,source=load_input(case,g,n);dt=float(np.median(np.diff(ts)))
    path=Path(path).resolve();q=np.load(path)['q'].copy();folder=path.parent/(path.stem+'_terminal_nullspace_v1')
    rank=rank_constant_nullspace(g,c,q,h);atomic_json(folder/'RANKING.json',dict(rows=rank,input=file_record(path),source=source))
    results=[];examined=0
    for i,row in enumerate(rank):
        candidate=q.copy();candidate[:,[6,13]]=row['wrist_joint_values']
        prefix=preparation_path(np.array(read(INITIAL)['g1_14_arm_initial_q_rad']),candidate[0],21)
        if not temporal_metrics(np.vstack((prefix[:-1],candidate)),dt,s.config)['pass_temporal']:continue
        out=folder/f'CANDIDATE_{i}.npz';atomic_npz(out,q=candidate,RAW_REPRESENTATION_TARGET=t,common_hand_q=h,source_timestamp=ts)
        qualify(case,str(out));p=folder/(out.stem+'_independent_qualification')/'FINAL_QUALIFICATION.json';r=read(p)
        results.append(dict(qualification=file_record(p),blocked=len(r['blocked_frames']),preparation_blocked=len(r['preparation_blocked_frames']),candidate=file_record(out)))
        examined+=1
        if r['qualified'] or examined>=4:break
    if results:
        best=min(results,key=lambda r:(r['preparation_blocked'],r['blocked'],r['candidate']['path']))
        atomic_json(folder/'SELECTED_CANDIDATE.json',dict(selected=best,results=results,selection='minimum geometry-invalid frames after unchanged fixed-prefix temporal qualification; no task outcome'))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
