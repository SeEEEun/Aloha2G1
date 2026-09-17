#!/usr/bin/env python3
"""Persist a preselected method-blind framewise oracle and diagnostic figure."""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.qualify_geometry_confirmed_position import RUN as BASELINE,CONTRACT as BASELINE_CONTRACT,verify,sha256
from tools.final_single_variable_prepare import OUT,RESET,read,file_record,status
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_text,atomic_npz,atomic_csv
from tools.doll_handoff_retargeting.common import load_common_config,load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.common_g1_morphology_adapter import CommonG1MorphologyAdapter
from tools.common_geometry_confirmed_collision import GeometryConfirmedCollision
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle

DEST=OUT/'02_common_execution_qualification/cartesian_reachability_forensic_v1'
CONTRACT=DEST/'ORACLE_CONTRACT.json'
CASES=(('WRIST',0),('WRIST',24),('WRIST',49),('INTERACTION',49))


def intervals(frames):
    result=[]
    for f in frames:
        if result and f==result[-1][1]+1:result[-1][1]=f
        else:result.append([f,f])
    return result


def model():
    common=load_common_config(RESET/'config/common_config.json');g1=G1Kinematics(common,load_scene(common))
    natural=np.asarray(common['resolved']['canonical_g1_nominal_q'])
    collision=GeometryConfirmedCollision(CommonG1MorphologyAdapter(common,g1,natural),1e-5)
    return g1,collision,natural


def prepare():
    verify(read(BASELINE_CONTRACT))
    if CONTRACT.exists():
        contract=read(CONTRACT)
        for record in contract['protected']+contract['implementations']:
            assert sha256(record['path'])==record['sha256'],record['path']
        return contract
    selected=[];protected=[file_record(BASELINE_CONTRACT)]
    for mode,ep in CASES:
        name=f'{mode}_EP{ep:03d}';r=read(BASELINE/(name+'.json'))
        with np.load(BASELINE/(name+'.npz'),allow_pickle=False) as z:
            error=z['position_residual_m'].max(axis=1);runs=intervals(r['failing_cartesian_frames']);chosen=set()
            for a,b in runs:chosen.update([a,(a+b)//2,b,a+int(np.argmax(error[a:b+1])),*range(a,b+1,10)])
        selected.append({'representation_mode':mode,'episode':ep,'name':name,'failing_intervals':runs,
                         'failed_frame_count':len(r['failing_cartesian_frames']),'sample_frames':sorted(chosen)})
        protected.extend(file_record(BASELINE/(name+suffix)) for suffix in ('.json','.npz'))
    cfg={'position_tolerance_m':.01,'seed_count':128,'maximum_evaluations_per_seed':180,
         'seed_policy':'natural, stand, joint midpoints, sixteen symmetric shoulder/elbow posture offsets, 109 deterministic unscrambled Halton full-limit seeds',
         'temporal_continuity_used':False,'previous_trajectory_seed_used':False,'target_projection_allowed':False,
         'stop_on_valid_witness':True,'unreachable_definition':'No collision/limit-valid 10mm witness in bounded common search; not proof of physical impossibility.'}
    g1,collision,natural=model();oracle=FramewiseReachabilityOracle(g1,collision,cfg,natural)
    code=[ROOT/'tools/common_framewise_reachability_oracle.py',Path(__file__)]
    tree=ast.parse(code[0].read_text());forbidden={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}&{'method','episode','representation_mode','task_success','object_pose','previous_q'}
    assert not forbidden
    contract={'oracle':cfg,'selected_before_oracle_outcomes':True,'fixed_sample_stride_frames':10,'cases':selected,
      'protected':protected,'implementations':[file_record(p) for p in code],'seed_vectors':oracle.seeds.tolist(),
      'collision_rule':'Unchanged frozen GeometryConfirmedCollision; HARD or UNRESOLVED blocks a witness.',
      'decision_rule':'All sampled failures reachable => SEQUENTIAL_IK_SOLVER_FAILURE. Both witnessed reachable and bounded no-witness => MIXED; densify any interval with no-witness samples before reporting exact counts. No-witness alone does not certify global infeasibility.',
      'minimum_correction_rule':'Report witnessed correction as an upper bound, never a globally minimal correction without a certified lower bound.',
      'environment':{'python':sys.executable,**{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}}
    for name in ('CURRENT_STATUS.md','CURRENT_STATUS.json'):
        if (OUT/name).exists():atomic_text(DEST/('PREVIOUS_'+name),(OUT/name).read_text())
    atomic_json(CONTRACT,contract)
    status('FRAMEWISE_ORACLE_CONTRACT_AND_SAMPLES_FROZEN','RUN_98_PRESELECTED_FRAMEWISE_ORACLE_SAMPLES',[CONTRACT])
    print('ORACLE_SELECTION',[(r['name'],len(r['sample_frames'])) for r in selected],flush=True)
    return contract


def run_samples():
    contract=prepare();g1,collision,natural=model();oracle=FramewiseReachabilityOracle(g1,collision,contract['oracle'],natural)
    for case in contract['cases']:
        with np.load(BASELINE/(case['name']+'.npz'),allow_pickle=False) as source:
            for frame in case['sample_frames']:
                path=DEST/'samples'/f"{case['name']}_F{frame:04d}.json"
                if path.exists():
                    result=read(path);assert result['contract_sha256']==sha256(CONTRACT)
                    print('REUSE',case['name'],frame,result['classification'],flush=True);continue
                target=source['RAW_REPRESENTATION_TARGET'][frame].copy();hands=source['common_hand_q'][frame].copy()
                result=oracle.solve(target,hands)
                result.update({'case':case['name'],'frame':frame,'timestamp_s':float(source['source_timestamp'][frame]),
                               'incoming_target_m':target,'common_hand_q':hands,'sequential_residual_mm':1000*float(source['position_residual_m'][frame].max()),
                               'contract_sha256':sha256(CONTRACT),'target_sha256':hashlib.sha256(target.tobytes()).hexdigest()})
                atomic_json(path,result)
                status('FRAMEWISE_ORACLE_SAMPLE_PERSISTED','REMAINING_PRESELECTED_ORACLE_SAMPLES',[CONTRACT,path])
                print(case['name'],frame,result['classification'],'best_mm',1000*result['best']['maximum_residual_m'],'seeds',result['seeds_tested'],flush=True)
    return summarize()


def summarize():
    contract=prepare();rows=[];by_case={}
    for case in contract['cases']:
        entries=[read(DEST/'samples'/f"{case['name']}_F{f:04d}.json") for f in case['sample_frames']]
        by_case[case['name']]=entries;rows.extend(entries)
    counts={name:{'reachable':sum(r['classification']=='FRAME_REACHABLE' for r in entries),
                  'unreachable_bounded':sum(r['classification']=='FRAME_UNREACHABLE' for r in entries),'sampled':len(entries)} for name,entries in by_case.items()}
    all_reachable=all(r['classification']=='FRAME_REACHABLE' for r in rows)
    cause='SEQUENTIAL_IK_SOLVER_FAILURE' if all_reachable else 'MIXED'
    result={'root_cause':cause,'counts':counts,'all_sampled_failures_reachable':all_reachable,'targets_modified':False,
            'global_unreachability_proven':False,'contract':file_record(CONTRACT),'rows':rows,
            'next_gate':'COMMON_SEQUENTIAL_RECOVERY' if all_reachable else 'DENSIFY_NO_WITNESS_INTERVALS; DO_NOT_ASSERT_GLOBAL_INFEASIBILITY'}
    atomic_json(DEST/'CARTESIAN_REACHABILITY_FORENSIC.json',result)
    atomic_csv(DEST/'ORACLE_SAMPLED_TARGETS.csv',[{'case':r['case'],'frame':r['frame'],'timestamp_s':r['timestamp_s'],'classification':r['classification'],
       'sequential_residual_mm':r['sequential_residual_mm'],'oracle_residual_mm':1000*r['best']['maximum_residual_m'],
       'seed_id':r['best']['seed_id'],'joint_limit_margin_rad':r['best']['joint_limit_margin_rad'],'geometry_valid':r['best']['geometry_valid'],'seeds_tested':r['seeds_tested']} for r in rows])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(13,10),sharex=False,constrained_layout=True)
    for ax,case in zip(axes,contract['cases']):
        entries=by_case[case['name']]
        with np.load(BASELINE/(case['name']+'.npz'),allow_pickle=False) as z:
            ax.plot(z['source_timestamp'],1000*z['position_residual_m'].max(axis=1),color='#c65136',lw=1.3,label='Sequential IK residual')
        for classification,color,marker in [('FRAME_REACHABLE','#087f5b','o'),('FRAME_UNREACHABLE','#8c2d9b','x')]:
            subset=[r for r in entries if r['classification']==classification]
            if subset:ax.scatter([r['timestamp_s'] for r in subset],[1000*r['best']['maximum_residual_m'] for r in subset],c=color,marker=marker,s=24,label='Oracle '+classification,zorder=4)
        ax.axhline(10,color='#303030',ls='--',lw=1,label='10 mm gate')
        ax.set_ylabel('Residual (mm)');ax.set_title(case['name'].replace('WRIST_EP','A').replace('INTERACTION_EP','B')+' — oracle samples only',loc='left')
        ax.set_ylim(bottom=-3);ax.grid(alpha=.2);ax.set_xlabel('Source time (s)');ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('Cartesian reachability forensic: unchanged targets, limits and collision rule',fontsize=14)
    for suffix in ('png','pdf','svg'):fig.savefig(DEST/f'CARTESIAN_REACHABILITY_FORENSIC.{suffix}',dpi=180)
    plt.close(fig)
    report='# Cartesian reachability forensic\n\n'+cause+'\n\n'
    report+='Frame-independent, deterministic common 128-seed budget, at most 180 evaluations per seed. Stop early only after a valid witness. Same frozen geometry-confirmed collision classifier, joint limits and 10 mm gate. No targets or timestamps changed.\n\n'
    report+='| Case | Reachable sampled failures | No witness within bounded search |\n|---|---:|---:|\n'
    for name,c in counts.items():report+=f"| {name} | {c['reachable']}/{c['sampled']} | {c['unreachable_bounded']}/{c['sampled']} |\n"
    report+='\nSampling: first/middle/last/worst residual per failing interval and every 10 frames; selection frozen before oracle outcomes. Per-target q, seed ID, residual, hard-limit margin and complete detailed collision records are in `samples/`. Unobserved frames are not labeled reachable or unreachable by interpolation. Bounded search failure is not a global unreachability proof; a found correction is only an upper bound on a minimum correction.\n'
    atomic_text(DEST/'CARTESIAN_REACHABILITY_FORENSIC.md',report)
    status(cause,result['next_gate'],[CONTRACT,DEST/'CARTESIAN_REACHABILITY_FORENSIC.json',DEST/'CARTESIAN_REACHABILITY_FORENSIC.md',DEST/'CARTESIAN_REACHABILITY_FORENSIC.png'])
    print('ORACLE_ROOT_CAUSE',cause,counts,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.summarize:summarize()
    else:run_samples()
