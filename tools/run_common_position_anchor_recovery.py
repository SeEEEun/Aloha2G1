#!/usr/bin/env python3
"""One fixed method-blind recovery attempt on all six unchanged smoke inputs."""
import argparse
import ast
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cartesian_reachability_forensic import prepare,model,DEST,BASELINE
from tools.final_single_variable_prepare import OUT,read,file_record,status
from tools.final_single_variable_qualify_position import CONFIG,evaluate
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_npz,atomic_text,sha256
from tools.common_position_anchor_recovery import AnchorRecoveryPositionSolver

REC=DEST/'common_anchor_recovery_v2'
PRIOR_REC=DEST/'common_anchor_recovery_v1'
CONTRACT=REC/'RECOVERY_CONTRACT.json'


def preflight():
    prepare()
    settings={'anchor_seed_budget':16,'anchor_posture_weight':.003,'executable_guide_weight':1.0,
              'strategy':'forward full-limit continuous posture anchors, then original temporally bounded execution; one deterministic full-limit anchor/reseed attempt when Cartesian residual exceeds the unchanged gate',
              'choose_anchor':'closest-to-previous valid configuration among generated candidates; residual-ranked when no valid Cartesian witness',
              'instantaneous_anchor_jump_allowed':False,'target_projection_allowed':False,'temporal_and_collision_thresholds_unchanged':True,
              'dominance_pruning':'Skip geometry evaluation only if the candidate cannot win even assuming it is collision-free; selected states and collision acceptance are unchanged.',
              'attempt_budget':'one common configuration, one guide pass, one executable pass, one bounded reseed per non-certified-failing frame'}
    code=[file_record(ROOT/'tools/common_position_anchor_recovery.py'),file_record(Path(__file__))]
    tree=ast.parse((ROOT/'tools/common_position_anchor_recovery.py').read_text())
    assert not ({n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}&{'method','episode','representation_mode','object_pose','task_success'})
    cfg={'settings':settings,'codes':code,'original_solver_config':file_record(CONFIG),'source_oracle_contract':file_record(DEST/'ORACLE_CONTRACT.json')}
    if CONTRACT.exists():assert read(CONTRACT)==cfg
    else:atomic_json(CONTRACT,cfg)
    return settings


def verify_dominance_equivalence(name):
    """Replay stored candidate ranks, proving pruning preserves every winner."""
    details=read(PRIOR_REC/(name+'_CANDIDATES.json'))
    records=details['anchor_reports']+[r['anchor'] for r in details['framewise_recovery_attempts']]
    skipped=0
    for record in records:
        full=None;pruned=None;full_seed=None;pruned_seed=None
        for row in record['candidates']:
            error=row['maximum_residual_m'];step=row['step_from_previous_rad']
            optimistic=(False,error>.01,step if error<=.01 else error,step)
            actual=(bool(row['contacts']),*optimistic[1:])
            if full is None or actual<full:full=actual;full_seed=row['seed_id']
            if pruned is not None and optimistic>=pruned:skipped+=1;continue
            if pruned is None or actual<pruned:pruned=actual;pruned_seed=row['seed_id']
        assert full==pruned and full_seed==pruned_seed
        assert tuple(record['selected_key'])==full
    return {'status':'PASS','anchor_records_checked':len(records),'provably_dominated_geometry_queries':skipped,
            'proof':'Actual candidate key is never better than its collision-free optimistic key; skipping only optimistic_key >= best_key preserves the strict-minimum winner.',
            'source_candidates':file_record(PRIOR_REC/(name+'_CANDIDATES.json'))}


def run(mode,ep):
    settings=preflight();base=REC/f'{mode}_EP{ep:03d}';result_path=base.with_suffix('.json')
    if result_path.exists():
        r=read(result_path);assert r['contract_sha256']==sha256(CONTRACT);return r
    prior=PRIOR_REC/(base.name+'.json')
    if prior.exists():
        equivalence=verify_dominance_equivalence(base.name)
        result=read(prior)
        assert sha256(Path(result['trajectory']['path']))==result['trajectory']['sha256']
        result['reused_equivalent_v1_result']=file_record(prior)
        result['original_contract_sha256']=result['contract_sha256'];result['contract_sha256']=sha256(CONTRACT)
        result['dominance_equivalence']=equivalence
        atomic_json(result_path,result)
        print('REUSE_EQUIVALENT_COMPLETED_TRAJECTORY',base.name,equivalence,flush=True)
        return result
    k,collision,natural=model();cfg=read(CONFIG);solver=AnchorRecoveryPositionSolver(k,collision,cfg,natural,settings)
    with np.load(BASELINE/(base.name+'.npz'),allow_pickle=False) as z:
        targets=z['RAW_REPRESENTATION_TARGET'].copy();hands=z['common_hand_q'].copy();timestamps=z['source_timestamp'].copy();initial=z['initial_q'].copy()
    print('COMMON_RECOVERY',base.name,'START',flush=True)
    q,details=solver.solve(targets,hands,timestamps,initial)
    metrics,actual,error,unused,margins=evaluate(solver,q,targets,hands,timestamps,cfg)
    metrics.pop('projection',None);metrics['position_qualified']=metrics['raw_position_qualified']
    frames=[]
    for frame in range(len(q)):
        records=collision.inspect(q[frame],*hands[frame]);frames.append({'frame':frame,'records':records})
    counts={c:sum(any(r['classification']==c for r in f['records']) for f in frames) for c in ('PROXY_ONLY_OVERLAP','HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')}
    atomic_npz(base.with_suffix('.npz'),q=q,RAW_REPRESENTATION_TARGET=targets,common_hand_q=hands,source_timestamp=timestamps,
               initial_q=initial,actual_wrist_position_model=actual,position_residual_m=error,posture_guide=details.pop('posture_guide'))
    atomic_json(base.with_name(base.name+'_CANDIDATES.json'),details);atomic_json(base.with_name(base.name+'_COLLISION_FRAMES.json'),frames)
    result={'representation_mode':mode,'episode':ep,'contract_sha256':sha256(CONTRACT),'classification_counts':counts,
            'runtime_s':details['runtime_s'],'trajectory':file_record(base.with_suffix('.npz')),'targets_modified':False,**metrics}
    atomic_json(result_path,result);preflight()
    atomic_json(REC/(base.name+'_STATUS.json'),{'completed':True,'result':file_record(result_path)})
    print('COMMON_RECOVERY_RESULT',base.name,{key:result[key] for key in ('position_qualified','raw_position_acceptance_rate','position_residual_mm','hard_limit_violations','branch_discontinuities','temporal_pass','classification_counts')},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['WRIST','INTERACTION']);p.add_argument('--episode',type=int,choices=[0,24,49]);p.add_argument('--all',action='store_true');a=p.parse_args()
    if a.all:
        for mode in ['WRIST','INTERACTION']:
            for ep in [0,24,49]:run(mode,ep)
    elif a.mode is not None and a.episode is not None:run(a.mode,a.episode)
    else:p.error('specify --all or --mode/--episode')
