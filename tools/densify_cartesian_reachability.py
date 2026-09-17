#!/usr/bin/env python3
"""Exhaust all originally failed frames with the same frozen framewise oracle."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cartesian_reachability_forensic import prepare,model,DEST,BASELINE,CONTRACT
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.common_g1_position_bounds import outer_enclosures,residual_lower_bounds
from tools.final_single_variable_prepare import read,file_record
from tools.run_reference_motion_scientific_reset import atomic_json,sha256


def run(name):
    contract=prepare();case=next(c for c in contract['cases'] if c['name']==name)
    g1,collision,natural=model();g1.assign(natural);enclosures=outer_enclosures(g1)
    oracle=FramewiseReachabilityOracle(g1,collision,contract['oracle'],natural)
    extension=DEST/'dense'/f'{name}_DENSE_CONTRACT.json'
    cfg={'parent_contract':file_record(CONTRACT),'case':case,'scope':'every originally failing frame, no interpolation of untested status',
         'enclosures':enclosures,'oracle_budget_unchanged':True,'codes':[file_record(Path(__file__)),file_record(ROOT/'tools/common_g1_position_bounds.py')],
         'certificate_numeric_guard_m':1e-8,'note':'Conservative outer-bound certificates are model-derived. Inside is not a reachability decision. Every frame still receives the full common oracle if no prior oracle witness exists.'}
    if extension.exists():
        old=read(extension)
        assert old==cfg,'dense contract changed'
    else:atomic_json(extension,cfg)
    all_frames=[f for a,b in case['failing_intervals'] for f in range(a,b+1)]
    with np.load(BASELINE/(name+'.npz'),allow_pickle=False) as z:
        for index,frame in enumerate(all_frames):
            path=DEST/'dense'/f'{name}_F{frame:04d}.json'
            if path.exists():
                assert read(path)['dense_contract_sha256']==sha256(extension)
                continue
            source=DEST/'samples'/path.name
            if frame in case['sample_frames']:
                if not source.exists():raise RuntimeError('predeclared sample not completed: '+str(source))
                result=read(source);origin=file_record(source)
            else:
                result=oracle.solve(z['RAW_REPRESENTATION_TARGET'][frame],z['common_hand_q'][frame]);origin=None
            lower=residual_lower_bounds(z['RAW_REPRESENTATION_TARGET'][frame],enclosures)
            certificate=bool(lower.max()>.01+cfg['certificate_numeric_guard_m'])
            if certificate:assert result['classification']!='FRAME_REACHABLE','outer bound contradicts FK witness'
            result.update({'case':name,'frame':frame,'timestamp_s':float(z['source_timestamp'][frame]),'target_sha256':hashlib.sha256(z['RAW_REPRESENTATION_TARGET'][frame].tobytes()).hexdigest(),
                'incoming_target_m':z['RAW_REPRESENTATION_TARGET'][frame],'common_hand_q':z['common_hand_q'][frame],
                'dense_contract_sha256':sha256(extension),'sequential_residual_mm':1000*float(z['position_residual_m'][frame].max()),
                'reused_predeclared_sample':origin,'certified_residual_lower_bound_m':lower.tolist(),
                'position_infeasibility_certified':certificate,'correction_to_gate_certified_lower_bound_m':max(float(lower.max())-.01,0.)})
            atomic_json(path,result)
            atomic_json(DEST/'dense'/f'{name}_PROGRESS.json',{'case':name,'completed':index+1,'total':len(all_frames),'last_result':file_record(path)})
            if index%10==0 or index+1==len(all_frames):print(name,index+1,'/',len(all_frames),frame,result['classification'],'certificate',certificate,flush=True)
    prepare()
    print(name,'DENSE_COMPLETE',len(all_frames),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['WRIST_EP000','WRIST_EP024','WRIST_EP049','INTERACTION_EP049']);run(p.parse_args().case)
