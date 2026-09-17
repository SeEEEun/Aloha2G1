#!/usr/bin/env python3
"""Separately versioned rerun: unchanged targets/IK, common geometry acceptance."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import platform
import subprocess
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.final_single_variable_prepare import OUT,RESET,read,file_record,status
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_npz,atomic_text,atomic_csv,sha256 as file_sha256
from tools.final_single_variable_qualify_position import CONFIG,evaluate,inspect_implementation
from tools.doll_handoff_retargeting.common import load_common_config,load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.common_g1_morphology_adapter import CommonG1MorphologyAdapter
from tools.final_common_position_solver import CommonPositionSolver
from tools.common_geometry_confirmed_collision import GeometryConfirmedCollision,BLOCKING,trimesh,rtree,MAX_TRIANGLE_PAIRS

QUAL=OUT/'02_common_execution_qualification'
RUN=QUAL/'position_only_geometry_confirmed_v2'
PREVIOUS=QUAL/'position_only'
CONTRACT=RUN/'COMMON_GEOMETRY_RULE_CONTRACT.json'
SMOKE=(0,24,49)
MODES=('WRIST','INTERACTION')


def sha256(path):
    return file_sha256(Path(path))


def identifiers(path):
    tree=ast.parse(Path(path).read_text())
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    return sorted(names&{'representation_mode','method','episode','object_pose','task_success','physical_outcome'})


def preflight():
    if CONTRACT.exists():
        contract=read(CONTRACT)
        verify(contract)
        return contract
    cfg=read(CONFIG)
    assert cfg['collision_penetration_tolerance_m']==1e-5
    # Preserve every prior artifact, plus all target/registration/config inputs.
    old_manifest=read(OUT/'ARTIFACT_HASH_MANIFEST.json')
    protected=[]
    for field in ('artifacts','implementations'):
        for item in old_manifest.get(field,[]):
            if Path(item['path']).name.startswith('CURRENT_STATUS'):continue
            assert sha256(item['path'])==item['sha256'],item['path']
            protected.append(file_record(item['path']))
    extra=[ROOT/'tools/common_g1_morphology_adapter.py',ROOT/'tools/doll_handoff_feasibility/solver.py',
           ROOT/'tools/doll_handoff_retargeting/models.py',ROOT/'tools/doll_handoff_retargeting/retarget.py',
           ROOT/'tools/reference_motion_common_timeline.py',ROOT/'configs/common_g1_morphology_adapter_v1.json',
           ROOT/'configs/doll_handoff_g1_feasibility_resolver.json',RESET/'config/common_config.json',RESET/'config/proposed_config.json',
           ROOT/'tools/final_common_position_solver.py',ROOT/'tools/final_single_variable_qualify_position.py',CONFIG]
    common=load_common_config(RESET/'config/common_config.json')
    extra.extend([Path(common['models']['g1_xml']),*sorted(Path(common['models']['g1_xml']).parent.glob('assets/*.STL'))])
    known={x['path'] for x in protected}
    protected.extend(file_record(p) for p in extra if str(p.resolve()) not in known)
    code=[ROOT/'tools/common_geometry_confirmed_collision.py',Path(__file__),ROOT/'tests/test_common_geometry_confirmed_collision.py']
    assert not identifiers(code[0])
    contract={'schema':'common_geometry_confirmation_v2','numerical_tolerance_m':cfg['collision_penetration_tolerance_m'],
      'tolerance_basis':'Unchanged existing collision_penetration_tolerance_m (10 micrometers). Prior raw/compiled transform disagreement <=9 nanometers; float32 mesh/float64 transforms. Not chosen from outcomes.',
      'proxy_broad_phase':'Unchanged CommonG1MorphologyAdapter._records, contact categories and 10-micrometer numerical tolerance.',
      'confirmation':'Exact compiled visual CAD triangles plus connected-shell containment. Open, non-oriented or degenerate shells receive conservative convex enclosures (degenerate fallback: enclosing box with each extent at least the same numerical tolerance). Enclosure-only ambiguity is UNRESOLVED; never hard or automatically clear.',
      'hard_rule':'Strict interior-depth witness > common tolerance between reliable closed shells.',
      'proxy_only_rule':'All original and conservatively enclosed components are separated beyond tolerance and are not contained in each other.',
      'unresolved_rule':'Unreliable enclosure overlap, near-surface ambiguity, missing geometry, numerical failure, or finite geometry-budget exhaustion fails closed.',
      'geometry_pair_budget':MAX_TRIANGLE_PAIRS,'representation_input_allowed':False,'pair_whitelist':[],
      'containment_queries':'32-point bounded-memory batches; two fixed bidirectional ray directions, no random fallback; ambiguous parity fails closed.',
      'previous_attempt':'v1 stopped for excessive unbatched containment working set before any completed trajectory; contract, code and 39-contact diagnostics preserved in sibling v1 directory.',
      'outcome_based_thresholds':False,'shared_left_right':True,'shared_modes':True,
      'position_acceptance':{'tolerance_m':cfg['position_tolerance_m'],'minimum_frame_acceptance':cfg['required_frame_acceptance_rate'],
                             'target_projection_applied':False,'note':'Use raw-target acceptance only; do not apply or qualify a shifted target.'},
      'unchanged_solver':'Exact prior CommonPositionSolver implementation, objective weights, seed strategy, budgets, hard limits and temporal settings. Only its collision-record provider is replaced.',
      'stop_gate':'Either method below 3/3 raw position acceptance stops orientation and Dex3.',
      'smoke_episodes':list(SMOKE),'protected_artifacts':protected,'new_implementations':[file_record(p) for p in code],
      'environment':{'python':sys.executable,'python_version':platform.python_version(),'numpy':np.__version__,'mujoco':mujoco.__version__,'trimesh':trimesh.__version__,'rtree':rtree.__version__}}
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_common_geometry_confirmed_collision.py','-v'],cwd=ROOT,text=True,capture_output=True)
    atomic_json(RUN/'GEOMETRY_RULE_UNIT_TESTS.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,'test_source':file_record(code[2])})
    if tests.returncode:raise RuntimeError(tests.stderr)
    for name in ('CURRENT_STATUS.md','CURRENT_STATUS.json'):
        path=OUT/name
        if path.exists():atomic_text(RUN/('PREVIOUS_'+name),path.read_text())
    atomic_json(CONTRACT,contract)
    status('COMMON_GEOMETRY_RULE_DEFINED','GEOMETRY_VALIDATION_AND_SIX_POSITION_RERUNS',[CONTRACT,*code])
    return contract


def audit_previous():
    contract=preflight();verify(contract)
    path=RUN/'PREVIOUS_PROXY_CONTACT_RECLASSIFICATION.json'
    if path.exists():return read(path)
    common=load_common_config(RESET/'config/common_config.json');g1=G1Kinematics(common,load_scene(common))
    proxy=CommonG1MorphologyAdapter(common,g1,np.array(common['resolved']['canonical_g1_nominal_q']))
    classifier=GeometryConfirmedCollision(proxy,contract['numerical_tolerance_m'])
    rows=[]
    for mode in MODES:
        for ep in SMOKE:
            old=read(PREVIOUS/f'{mode}_EP{ep:03d}.json')
            with np.load(PREVIOUS/f'{mode}_EP{ep:03d}.npz',allow_pickle=False) as z:
                for row in old['contacts']:
                    f=row['frame']
                    rows.append({'representation_mode':mode,'episode':ep,'frame':f,'records':classifier.inspect(z['q'][f],*z['common_hand_q'][f])})
    result={'contract_sha256':sha256(CONTRACT),'rows':rows,'counts':{c:sum(any(r['classification']==c for r in f['records']) for f in rows) for c in ('PROXY_ONLY_OVERLAP','HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')}}
    atomic_json(path,result);verify(contract)
    print('PRIOR_CONTACT_RECLASSIFICATION',result['counts'],flush=True)
    return result


def verify(contract):
    mismatches=[p['path'] for p in contract['protected_artifacts']+contract['new_implementations'] if sha256(p['path'])!=p['sha256']]
    if mismatches:raise RuntimeError('Frozen rerun dependencies changed: '+repr(mismatches))


def compress_frames(frames):
    runs=[]
    for f in frames:
        if runs and f==runs[-1][1]+1:runs[-1][1]=f
        else:runs.append([f,f])
    return ', '.join(str(a) if a==b else f'{a}–{b}' for a,b in runs) or 'none'


def run(episode,mode):
    contract=preflight();verify(contract);cfg=read(CONFIG)
    base=RUN/f'{mode}_EP{episode:03d}';result_path=base.with_suffix('.json')
    if result_path.exists():
        result=read(result_path)
        assert result['contract_sha256']==sha256(CONTRACT)
        for item in result['artifacts']:assert sha256(item['path'])==item['sha256']
        print('REUSE',mode,episode,result['position_qualified'],flush=True);return result
    raw=OUT/'01_registration/raw_references'/f'TRAIN_EP{episode:03d}.npz'
    common=load_common_config(RESET/'config/common_config.json');scene=load_scene(common)
    g1=G1Kinematics(common,scene);nominal=np.array(common['resolved']['canonical_g1_nominal_q'])
    initial=np.array(read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad'])
    proxy=CommonG1MorphologyAdapter(common,g1,nominal)
    collision=GeometryConfirmedCollision(proxy,cfg['collision_penetration_tolerance_m'])
    primitives=g1.derive_hand_primitives(scene,read(RESET/'config/proposed_config.json'),nominal)
    with np.load(raw,allow_pickle=False) as source:
        targets=np.stack([source[f'{mode}_{side}_wrist_position_model'] for side in ('left','right')],axis=1)
        hands=np.stack([np.asarray(primitives['states'][side]['OPEN'])+source[f'common_{side}_close_fraction'][:,None]*(np.asarray(primitives['states'][side]['GRASP'])-np.asarray(primitives['states'][side]['OPEN'])) for side in ('left','right')],axis=1)
        timestamps=source['source_timestamp'].copy()
    with np.load((PREVIOUS/base.name).with_suffix('.npz'),allow_pickle=False) as old:
        assert np.array_equal(targets,old['RAW_REPRESENTATION_TARGET'])
        assert np.array_equal(hands,old['common_hand_q'])
        assert np.array_equal(timestamps,old['source_timestamp'])
        old_q=old['q'].copy()
    solver=CommonPositionSolver(g1,collision,cfg,nominal)
    print(f'GEOMETRY POSITION {mode} {episode} START',flush=True);started=time.monotonic()
    q,details=solver.solve(targets,hands,timestamps,initial)
    metrics,actual,residual,unused_candidate,margins=evaluate(solver,q,targets,hands,timestamps,cfg)
    # The old evaluator calculates a diagnostic projection. It is not applied,
    # saved as a corrected target, or accepted in this strictly raw-target rerun.
    metrics.pop('projection',None)
    metrics['position_qualified']=bool(metrics['raw_position_qualified'])
    frames=[];residual_rows=[]
    for frame in range(len(q)):
        records=collision.inspect(q[frame],*hands[frame])
        classes={r['classification'] for r in records}
        classification=next((c for c in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY','PROXY_ONLY_OVERLAP') if c in classes),'CLEAR')
        distances=[r['detailed_separation_mm'] for r in records if r.get('detailed_separation_mm') is not None]
        frames.append({'frame':frame,'timestamp_s':float(timestamps[frame]),'proxy_collision':bool(records),
          'proxy_penetration_mm':max((r['proxy_penetration_mm'] for r in records),default=0.),
          'detailed_geometry_checked':bool(records),'detailed_collision':True if 'HARD_SELF_COLLISION' in classes else None if 'UNRESOLVED_GEOMETRY' in classes else False,
          'detailed_separation_mm':min(distances) if distances else None,'classification':classification,'records':records})
        if frame in metrics['failing_cartesian_frames']:
            residual_rows.append({'episode':episode,'representation_mode':mode,'frame':frame,'timestamp_s':float(timestamps[frame]),
              'left_residual_mm':1000*float(residual[frame,0]),'right_residual_mm':1000*float(residual[frame,1]),
              'maximum_residual_mm':1000*float(residual[frame].max()),'tolerance_mm':1000*cfg['position_tolerance_m']})
    counts={key:sum(any(r['classification']==key for r in f['records']) for f in frames) for key in ('PROXY_ONLY_OVERLAP','HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')}
    reasons=[]
    if metrics['raw_position_acceptance_rate']<cfg['required_frame_acceptance_rate']:reasons.append('CARTESIAN_RESIDUAL')
    if counts['HARD_SELF_COLLISION']:reasons.append('HARD_SELF_COLLISION')
    if counts['UNRESOLVED_GEOMETRY']:reasons.append('UNRESOLVED_GEOMETRY')
    if metrics['hard_limit_violations']:reasons.append('JOINT_LIMIT')
    if not metrics['temporal_pass'] or metrics['branch_discontinuities']:reasons.append('TEMPORAL_CONTINUITY')
    if not metrics['finite']:reasons.append('SOLVER_NUMERICAL')
    assert metrics['position_qualified']==(not reasons)
    atomic_npz(base.with_suffix('.npz'),q=q,source_timestamp=timestamps,RAW_REPRESENTATION_TARGET=targets,
      common_hand_q=hands,initial_q=initial,actual_wrist_position_model=actual,position_residual_m=residual,
      proxy_collision_margin_m=margins,reverse_posture_guide=details.pop('reverse_guide'))
    paths=[base.with_suffix('.npz'),base.with_name(base.name+'_CANDIDATES.json'),base.with_name(base.name+'_COLLISION_FRAMES.json'),
           base.with_name(base.name+'_COLLISION_FRAMES.csv'),base.with_name(base.name+'_RESIDUAL_FRAMES.csv')]
    atomic_json(paths[1],details);atomic_json(paths[2],frames)
    atomic_csv(paths[3],[{k:v for k,v in f.items() if k!='records'} for f in frames])
    atomic_csv(paths[4],residual_rows,fieldnames=['episode','representation_mode','frame','timestamp_s','left_residual_mm','right_residual_mm','maximum_residual_mm','tolerance_mm'])
    result={'representation_mode':mode,'episode_index':episode,'contract_sha256':sha256(CONTRACT),'raw_source':file_record(raw),
      'targets_identical_to_previous':True,'hands_identical_to_previous':True,'timestamps_identical_to_previous':True,
      'target_array_sha256':hashlib.sha256(targets.tobytes()).hexdigest(),'q_maximum_change_from_previous_rad':float(np.max(np.abs(q-old_q))),
      'classification_counts':counts,'failure_reasons':reasons,'failing_frame_intervals':compress_frames(metrics['failing_cartesian_frames']),
      'solver_runtime_s':details['runtime_s'],'total_runtime_s':time.monotonic()-started,'geometry_queries':collision.calls,
      'mesh_inventory':collision.mesh_inventory,'solver_implementation_audit':inspect_implementation(),'artifacts':[file_record(p) for p in paths],**metrics}
    atomic_json(result_path,result);verify(contract)
    status(f'GEOMETRY_POSITION_{mode}_EP{episode:03d}_PERSISTED','REMAINING_MATCHED_POSITION_SMOKE',[CONTRACT,result_path,*paths])
    print(json.dumps({k:result[k] for k in ('representation_mode','episode_index','position_qualified','classification_counts','failure_reasons','raw_position_acceptance_rate','position_residual_mm','solver_runtime_s')}),flush=True)
    return result


def summarize():
    contract=preflight();verify(contract)
    rows=[read(RUN/f'{mode}_EP{ep:03d}.json') for mode in MODES for ep in SMOKE]
    counts={mode:sum(r['position_qualified'] for r in rows if r['representation_mode']==mode) for mode in MODES}
    collision={mode:{kind:sum(r['classification_counts'][kind] for r in rows if r['representation_mode']==mode) for kind in ('PROXY_ONLY_OVERLAP','HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')} for mode in MODES}
    cartesian=any('CARTESIAN_RESIDUAL' in r['failure_reasons'] for r in rows)
    complete=all(r['position_qualified'] for r in rows)
    terminal='COMMON_POSITION_IK_QUALIFIED_AFTER_GEOMETRY_AUDIT' if complete else 'COMMON_CARTESIAN_IK_STILL_INVALID' if cartesian else 'COMMON_HARD_SELF_COLLISION_STILL_INVALID'
    summary={'status':terminal,'position_counts':counts,'collision_counts':collision,'rows':rows,'common_execution_identical':True,
      'full_6d':'NOT_RUN','loaded_dex3':'NOT_RUN','ready_for_dataset_finalization':False,'geometry_rule_contract':file_record(CONTRACT),
      'orientation_next_gate_permitted':complete,'protected_artifact_mismatches':[]}
    summary['prior_contact_audit']=file_record(RUN/'PREVIOUS_PROXY_CONTACT_RECLASSIFICATION.json')
    summary['unit_tests']=file_record(RUN/'GEOMETRY_RULE_UNIT_TESTS.json')
    atomic_json(QUAL/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.json',summary)
    text='# Geometry-confirmed common collision audit\n\n'+terminal+'\n\n'
    text+='One common numerical tolerance: **10 μm**, unchanged from the original collision infrastructure. No 2 mm allowance, pair whitelist, side-specific rule or representation input.\n\n'
    text+='Existing proxies nominate contacts. Original compiled CAD triangle surfaces and solid containment confirm them. For topology-defective shells, conservative convex enclosures may certify separation; their overlap is UNRESOLVED, not hard and not automatically clear. Enclosures do not alter robot/model assets or solver targets. All details, including every CLEAR frame and proxy-only contact, are retained in per-trajectory COLLISION_FRAMES JSON/CSV files. CLEAR frames were not narrow-phase queried; their detailed separation is null, not an invented distance.\n\n'
    text+='| Mode | Proxy-only frames | Hard frames | Unresolved frames |\n|---|---:|---:|---:|\n'
    for mode in MODES:
        c=collision[mode];text+=f"| {mode} | {c['PROXY_ONLY_OVERLAP']} | {c['HARD_SELF_COLLISION']} | {c['UNRESOLVED_GEOMETRY']} |\n"
    text+='\nFixed rule contract and hashes: `position_only_geometry_confirmed_v2/COMMON_GEOMETRY_RULE_CONTRACT.json`. Existing solver, targets, hand commands, timestamps and protected provenance remain hash-identical. Geometry-classifier and runner hashes were fixed before the six reruns.\n'
    atomic_text(QUAL/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.md',text)
    text='# Position IK after common geometry-confirmed collision rule\n\n'+terminal+'\n\n'
    text+='Cartesian gate remains **10 mm at ≥95% of frames**, exactly as predeclared. No targets are regenerated, projected, or replaced. Every over-tolerance frame is listed below, including those within the existing 5% allowance. All other gates require zero failures.\n\n'
    text+='| Mode | Episode | Accepted | Residual mean/p95/max (mm) | Proxy-only | Hard | Unresolved | Limits | Branches | Qualified | Reasons |\n|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|\n'
    for r in rows:
        p=r['position_residual_mm'];c=r['classification_counts']
        text+=f"| {r['representation_mode']} | {r['episode_index']} | {r['raw_position_acceptance_rate']:.6%} | {p['mean']:.6f}/{p['p95']:.6f}/{p['max']:.6f} | {c['PROXY_ONLY_OVERLAP']} | {c['HARD_SELF_COLLISION']} | {c['UNRESOLVED_GEOMETRY']} | {r['hard_limit_violations']} | {r['branch_discontinuities']} | {r['position_qualified']} | {', '.join(r['failure_reasons']) or 'NONE'} |\n"
    for r in rows:
        text+=f"\n## {r['representation_mode']} episode {r['episode_index']}\n\nFailure reasons: {', '.join(r['failure_reasons']) or 'NONE'}. Exact over-tolerance frames: {r['failing_frame_intervals']}.\n\n"
        text+=f"qdot max: {r['maximum_velocity_rad_s']:.9f} rad/s; qddot max: {r['maximum_acceleration_rad_s2']:.9f} rad/s²; temporal pass: {r['temporal_pass']}; finite: {r['finite']}; runtime: {r['solver_runtime_s']:.3f} s.\n\n"
        text+='| Frame | Timestamp (s) | Left residual (mm) | Right residual (mm) |\n|---:|---:|---:|---:|\n'
        with (RUN/f"{r['representation_mode']}_EP{r['episode_index']:03d}_RESIDUAL_FRAMES.csv").open() as f:
            for x in csv.DictReader(f):text+=f"| {x['frame']} | {float(x['timestamp_s']):.9f} | {float(x['left_residual_mm']):.9f} | {float(x['right_residual_mm']):.9f} |\n"
    text+='\nThis bounded solve is not a proof of globally UNREACHABLE_POSITION. No orientation, loaded Dex3, training, or PhysX execution is authorized past a failed Cartesian gate.\n'
    atomic_text(QUAL/'POSITION_IK_AFTER_COLLISION_RULE.md',text)
    parity='# Final common pipeline parity audit\n\nPASS — implemented smoke execution is common; qualification remains separate.\n\n'
    parity+='A/B target generation remains the only intended difference: WRIST versus INTERACTION. All six runs use the same byte-identical prior solver and configuration, new shared geometry classifier, morphology adapter, model, hard limits, seed/continuity strategy, initial state, hand commands and timestamp convention. Targets/hand commands/timestamps were checked numerically against each prior trajectory; all protected artifacts were SHA256-verified before and after execution. No representation identity reaches the solver or collision classifier. No pair whitelist, method-specific tolerance, workspace change, or target projection was introduced.\n\n'
    parity+=f"Position qualification: WRIST {counts['WRIST']}/3; INTERACTION {counts['INTERACTION']}/3. Full 6D and loaded Dex3: NOT RUN. Dataset/training parity and downstream physical execution are not newly qualified by this audit.\n\n"
    parity+='See `02_common_execution_qualification/position_only_geometry_confirmed_v2/COMMON_GEOMETRY_RULE_CONTRACT.json` for protected/new implementation hashes and the exact shared tolerance.\n'
    atomic_text(OUT/'FINAL_COMMON_PIPELINE_PARITY_AUDIT.md',parity)
    paths=[QUAL/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.json',QUAL/'GEOMETRY_CONFIRMED_COLLISION_AUDIT.md',QUAL/'POSITION_IK_AFTER_COLLISION_RULE.md',OUT/'FINAL_COMMON_PIPELINE_PARITY_AUDIT.md']
    atomic_json(RUN/'COMPLETION_HASH_MANIFEST.json',{'contract':file_record(CONTRACT),'reports':[file_record(p) for p in paths],'trajectories':[file_record(RUN/f'{mode}_EP{ep:03d}.json') for mode in MODES for ep in SMOKE]})
    status(terminal,'COMMON_ORIENTATION_CONVENTION_AUDIT' if complete else 'STOP_CARTESIAN_GATE_FAILED; NO_ORIENTATION_OR_DEX3',[CONTRACT,*paths,RUN/'COMPLETION_HASH_MANIFEST.json'])
    print(json.dumps({k:summary[k] for k in ('status','position_counts','collision_counts','full_6d','loaded_dex3','common_execution_identical','ready_for_dataset_finalization')}),flush=True)
    for r in rows:print(f"{r['representation_mode']} EP{r['episode_index']:03d} {r['failure_reasons']}: {r['failing_frame_intervals']}",flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--audit-previous',action='store_true');parser.add_argument('--all',action='store_true');parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--episode',type=int,choices=SMOKE);parser.add_argument('--mode',choices=MODES)
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.audit_previous:audit_previous()
    elif args.all:
        audit_previous()
        for mode in MODES:
            for ep in SMOKE:run(ep,mode)
        summarize()
    elif args.summarize:summarize()
    elif args.mode is not None and args.episode is not None:run(args.episode,args.mode)
    else:parser.error('specify --preflight, --all, --summarize, or --episode/--mode')
