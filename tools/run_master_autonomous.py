#!/usr/bin/env python3
"""Versioned, resumable orchestration. Labels only select frozen input files."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.master_autonomous_provenance import verified_oracle_contract
from tools.master_continuation_preflight import active_predecessors
from tools.cartesian_reachability_forensic import DEST,BASELINE,model
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.final_single_variable_prepare import OUT,RESET,read,file_record
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_text,atomic_npz,atomic_csv
from tools.doll_handoff_retargeting.common import load_scene
from tools.master_autonomous_common import preparation_duration,preparation_path,temporal_metrics

MASTER=OUT/'master_autonomous'; RUN=OUT/'02_common_execution_qualification/common_executable_position_v3'
QUAL=OUT/'01_registration/POSITION_SOLVER_QUALIFICATION_CONTRACT.json'
INITIAL=ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json'


def log(stage,status,inputs,problem,cause,action,artifacts,next_stage,korean,retries=0):
    row=dict(timestamp=datetime.now(timezone.utc).isoformat(),stage=stage,status=status,
        authoritative_inputs=[file_record(Path(p)) for p in inputs],observed_problem=problem,
        root_cause_classification=cause,action_taken=action,retries=retries,
        artifacts_created=[file_record(Path(p)) for p in artifacts],next_stage=next_stage)
    path=MASTER/'MASTER_RUN_LOG.jsonl';previous=path.read_text() if path.exists() else ''
    atomic_text(path,previous+json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    atomic_text(MASTER/'CURRENT_STAGE.md',f'# Autonomous execution\n\nStage: {stage}\n\nStatus: {status}\n\nNext: {next_stage}\n\n{action}\n')
    atomic_json(MASTER/'CHECKPOINT_STATE.json',row)
    atomic_text(MASTER/'CHATGPT_UPDATE.md',korean+'\n');print(korean,flush=True)


def initialize():
    assert not active_predecessors(),'Do not duplicate active predecessors'
    oracle,substitutions=verified_oracle_contract()
    manifest=read(DEST/'FINAL_HASH_MANIFEST.json')
    records=manifest['artifacts']+manifest['dense_files']+manifest['recovery_files']
    assert all(file_record(Path(r['path']))==r for r in records)
    for source in (OUT/'CURRENT_STATUS.md',OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md'):
        archive=MASTER/'provenance'/source.name
        if not archive.exists():atomic_text(archive,source.read_text())
    split=read(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json')
    contract=dict(schema='common_startup_and_executable_position_v3',training_ids=split['selection']['qualification_ids'],
        source= file_record(OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json'),
        old_limits=file_record(QUAL),initial_state=file_record(INITIAL),
        source_task_targets_unchanged=True,source_relative_event_times_unchanged=True,
        global_frame_zero='exact common natural arm q0; common OPEN hands; no object manipulation',
        source_clock='original source timestamps retained; execution time = source elapsed + common preparation duration',
        preparation='common rest-to-rest quintic joint path; no grasp, lift or handoff events; only arm motion',
        duration='max analytical minimum over TRAIN11 x both representations, rounded up to common 30Hz grid; no padding',
        duration_scope='minimum within declared path family, not globally time-optimal',
        source_position_gate_m=.01,collision_tolerance_m=1e-5,
        certified_unreachable='preserve raw target; report actual FK and certified lower bound separately',
        uncertified_no_witness='additional bounded common search; never promote to unreachable without certificate',
        numerical_slack='deterministic TRAIN repeatability; cap 2mm; geometric relaxation looseness is NOT numerical repeatability',
        candidate_budget={'propagated_paths':4,'sparse_refinement_weights':[.003,.001,.0003],
                          'maximum_evaluations_per_refinement':100,'candidate_geometry_checks':'final candidate and bounded common collision repair'},
        method_blind_core=file_record(ROOT/'tools/master_autonomous_common.py'),
        supersedes='only the initial boundary, old 95% raw gate, and old 10mm projection cap; current user explicitly authorizes these common amendments',
        downstream_no_outcome_inputs=True)
    path=MASTER/'COMMON_STARTUP_EXECUTION_CONTRACT.json'
    if path.exists():assert read(path)==contract,'Use a new version for a changed contract'
    else:atomic_json(path,contract)
    atomic_json(MASTER/'INGEST_VERIFICATION.json',dict(verified_predecessor_records=len(records),report_only_substitutions=substitutions,
        dense_rerun=False,forensic=file_record(DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json')))
    atomic_text(MASTER/'BLOCKERS/INITIAL_BOUNDARY_RESOLVED.md','# Previous initial-boundary conflict — authorized common resolution\n\nThe current user explicitly separates GLOBAL frame zero from SOURCE task start using one common preparation prefix. Natural q0, raw targets, source-relative timing and collision/limit rules remain unchanged. Preparation is excluded only from representation tracking, not physical validity. The old blocker is no longer a stop condition.\n')
    atomic_text(MASTER/'LATEST_BLOCKER.md','Previous initial-boundary conflict: RESOLVED BY AUTHORIZED COMMON PREPARATION. New executable trajectories remain to be qualified.\n')
    log('FORENSIC_INGEST_AND_STARTUP_CONTRACT','PASS',[DEST/'FINAL_HASH_MANIFEST.json',QUAL],
        'Old initial boundary conflated global start and source task start','MIXED_SOLVER_AND_TARGET_FEASIBILITY',
        'Verified 634 prior artifacts; reused dense forensic; declared common preparation and explicit source tracking mask.',
        [path,MASTER/'INGEST_VERIFICATION.json'],'TRAIN11_STARTUP_CALIBRATION',
        '기존 포렌식 634개 해시를 검증했습니다. 사용자 승인에 따라 자연 초기상태와 소스 작업 시작을 공통 준비 구간으로 분리합니다. 타깃·등록·충돌 규칙은 유지되며 A/B 공정성에 영향 없이 CLI가 계속 실행됩니다.')
    return contract,oracle


def calibrate_startup():
    contract,oc=initialize();cfg=read(QUAL);g1,collision,natural=model()
    oracle=FramewiseReachabilityOracle(g1,collision,oc['oracle'],natural)
    initial=np.array(read(INITIAL)['g1_14_arm_initial_q_rad'])
    primitives=g1.derive_hand_primitives(load_scene(g1.common) if hasattr(g1,'common') else __import__('tools.doll_handoff_retargeting.common',fromlist=['load_common_config']).load_scene(__import__('tools.doll_handoff_retargeting.common',fromlist=['load_common_config']).load_common_config(RESET/'config/common_config.json')),read(RESET/'config/proposed_config.json'),natural)
    opened=np.array([primitives['states'][s]['OPEN'] for s in ('left','right')]);dt=1/30
    initial_records=collision.inspect(initial,*opened)
    assert not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in initial_records)
    rows=[]
    for ep in contract['training_ids']:
        raw=OUT/'01_registration/raw_references'/f'TRAIN_EP{ep:03d}.npz'
        with np.load(raw) as z:
            for mode in ('WRIST','INTERACTION'):
                name=f'{mode}_EP{ep:03d}';p=RUN/'startup'/f'{name}.json'
                if p.exists():rows.append(read(p));continue
                target=np.stack([z[f'{mode}_{s}_wrist_position_model'][0] for s in ('left','right')])
                witness=oracle.solve(target,opened)
                assert witness['classification']=='FRAME_REACHABLE'
                end=np.array(witness['best']['q']);n,q,duration=preparation_duration(initial,end,dt,cfg)
                geometry=[collision.inspect(v,*opened) for v in q]
                blocked=[i for i,rr in enumerate(geometry) if any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in rr)]
                row=dict(case=name,source=file_record(raw),target=target.tolist(),first_task_q=end.tolist(),
                    minimum_path_duration_s=duration,required_intervals=n,geometry=geometry,blocked_frames=blocked,
                    witness=witness,temporal=temporal_metrics(q,dt,cfg))
                atomic_json(p,row);rows.append(row);print('STARTUP',name,n,'blocked',len(blocked),flush=True)
    assert not any(r['blocked_frames'] for r in rows),'Common startup collision planning required'
    n=max(r['required_intervals'] for r in rows);full=[]
    for row in rows:
        q=preparation_path(initial,np.array(row['first_task_q']),n)
        geometry=[collision.inspect(v,*opened) for v in q]
        blocked=[i for i,rr in enumerate(geometry) if any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in rr)]
        temporal=temporal_metrics(q,dt,cfg)
        assert not blocked and temporal['pass_temporal']
        np.testing.assert_array_equal(q[0],initial)
        atomic_npz(RUN/'startup'/f"{row['case']}_COMMON_PREFIX.npz",q=q,common_hand_q=np.repeat(opened[None],len(q),axis=0),execution_timestamp=np.arange(n+1)*dt,raw_tracking_mask=np.zeros(n+1,dtype=bool))
        full.append(dict(case=row['case'],temporal=temporal,hard_collision_frames=0,unresolved_frames=0,geometry=geometry))
    result=dict(status='TRAIN11_PREPARATION_PATHS_QUALIFIED_SOURCE_JOIN_PENDING',PREP_DURATION_SECONDS=n*dt,
        PREP_NUM_FRAMES=n,endpoint_sample_count=n+1,PREP_QDOT_MAX=cfg['maximum_velocity_rad_s'],
        PREP_QDDOT_MAX=cfg['maximum_acceleration_rad_s2'],natural_q0_preserved=True,
        preparation_rule_identical=True,rows=rows,common_duration_validation=full,
        caveat='First-task configurations may need common branch continuation; if changed, recalibrate all TRAIN11 paths before final duration freeze. Full source join not yet qualified.',
        implementation=file_record(ROOT/'tools/master_autonomous_common.py'),contract=file_record(MASTER/'COMMON_STARTUP_EXECUTION_CONTRACT.json'))
    atomic_json(MASTER/'STARTUP_CALIBRATION.json',result)
    atomic_text(MASTER/'STARTUP_CALIBRATION.md',f'# Common TRAIN11 preparation calibration\n\n22/22 sampled paths valid. Natural q0 exact. Provisional duration {n*dt:.9f}s ({n} intervals at 30Hz). No source targets modified. No interaction events in prefix. Source join remains unqualified. Recalibrate identically if trajectory continuation changes task-start configurations; freeze only after qualification.\n')
    log('TRAIN11_STARTUP_CALIBRATION','PATHS_PASS_JOIN_PENDING',[MASTER/'COMMON_STARTUP_EXECUTION_CONTRACT.json'],
        'Source trajectories still need common executable recovery','SEQUENTIAL_AND_MORPHOLOGY_REALIZATION_PENDING',
        'Calibrated common rest-to-rest preparation over all 22 TRAIN11 first targets, checked every sampled frame.',
        [MASTER/'STARTUP_CALIBRATION.json',MASTER/'STARTUP_CALIBRATION.md'],'COMMON_POSITION_RECOVERY',
        f'공통 준비 경로 22/22가 자연 q0·충돌·속도·가속도 검증을 통과했습니다. 잠정 공통 길이는 {n*dt:.3f}초이며 소스 궤적 접합 검증 전에는 최종 동결하지 않습니다. 동일한 위치 솔버 복구를 계속 진행합니다.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','startup']);args=p.parse_args()
    initialize() if args.stage=='initialize' else calibrate_startup()
