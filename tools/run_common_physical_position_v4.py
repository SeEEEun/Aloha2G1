#!/usr/bin/env python3
"""Versioned common acceptance amendment, requalification and geometry repair."""
from pathlib import Path
import sys,argparse,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_autonomous_dual_position import *
from tools.common_physical_position_v4 import qualify_numeric,restore
from tools.master_autonomous_diagnostics import choose_candidate
from tools.run_reference_motion_scientific_reset import atomic_text

V4=OUT/'02_common_execution_qualification/common_physical_position_v4'
STAGE=MASTER/'physical_acceptance_v4'

def initialize():
    verified_oracle_contract()
    archive=STAGE/'previous_status'
    for p in [OUT/'CURRENT_STATUS.md',OUT/'CURRENT_STATUS.json',OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md',MASTER/'CURRENT_STAGE.md',MASTER/'CHATGPT_UPDATE.md',MASTER/'LATEST_BLOCKER.md',MASTER/'CHECKPOINT_STATE.json',MASTER/'FINAL_RESULT.json']:
        ap=archive/p.parent.name/p.name
        if p.exists() and not ap.exists():atomic_text(ap,p.read_text())
    commands=[['git','log','--all','--oneline','-S','0.179','--','tools','configs'],
        ['rg','-n','0\\.179|maximum_step_norm_rad','configs','tools','docs',str(OUT/'00_contract'),str(OUT/'01_registration'),'--glob','*.py','--glob','*.json','--glob','*.md']]
    evidence=[dict(command=cmd,returncode=(r:=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)).returncode,stdout=r.stdout,stderr=r.stderr) for cmd in commands]
    contract=dict(status='UNIFIED_PHYSICAL_TEMPORAL_ACCEPTANCE',authorization='User2026-09-06:0.179rad is internal only absent a provenance-backed physical derivation',
        physical_predeclared_aggregate_derivation_found=False,aggregate_step_classification='INTERNAL_SOLVER_STEP_TRUST_REGION_ONLY',
        old_configuration_preserved=file_record(QUAL),velocity_rad_s=read(QUAL)['maximum_velocity_rad_s'],acceleration_rad_s2=read(QUAL)['maximum_acceleration_rad_s2'],
        branch_absolute=read(QUAL)['branch_absolute_step_norm_rad'],branch_local_multiplier=read(QUAL)['branch_local_multiplier'],
        hard_limits_unchanged=True,raw_targets_unchanged=True,registration_unchanged=True,event_clock_unchanged=True,
        finite_required=True,detailed_collision_tolerance_m=1e-5,unresolved_geometry_fails_closed=True,
        closest_feasible_slack=file_record(RUN/'orbit_certificate/REPEATABILITY_RESULT.json'),
        per_joint_limits_not_relaxed=True,method_blind=True,search_evidence=evidence,
        source_code_evidence=[file_record(ROOT/'tools'/p) for p in ('final_common_position_solver.py','final_single_variable_qualify_position.py','doll_handoff_retargeting/common.py')],
        implementation=file_record(ROOT/'tools/common_physical_position_v4.py'))
    p=STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'
    if not p.exists():atomic_json(p,contract)
    text='#0.179rad provenance and unified common temporal acceptance\n\nNo explicit physical/predeclared acceptance derivation was found. The two declarative occurrences are solver/qualification configurations; no equation connects0.179rad to authoritative joint velocities or accelerations. Git pickaxe history provides no derivation. Other report hits are subsequent diagnostics or incidental numeric measurements. The original evaluator uses per-joint step/velocity/acceleration and adaptive branch checks, not a fixed aggregate rejection.\n\nUnder the user2026-09-06 instruction,0.179rad is INTERNAL_SOLVER_STEP_TRUST_REGION_ONLY. It is logged but not a final trajectory gate. Common acceptance retains4.5rad/s per joint,130rad/s² per joint, adaptive branch_flags(0.18,8), actual hard limits, finite states and the unchanged10μm geometry-confirmed rule. Geometry unresolved cases fail closed. Raw targets, registration, source event timing and the TRAIN-derived closest-feasible numerical slack are unchanged.\n\nAll legacy files remain immutable provenance. New versioned evaluation applies the same function to both methods. Promoting a numeric candidate is not a collision qualification pass.\n'
    if not (STAGE/'PROVENANCE_AUDIT.md').exists():atomic_text(STAGE/'PROVENANCE_AUDIT.md',text)
    log('UNIFIED_PHYSICAL_ACCEPTANCE','IN_PROGRESS',[p], 'Old aggregate-step semantics resolved by explicit user instruction after provenance audit',
        'INTERNAL_SOLVER_TRUST_REGION_NOT_PHYSICAL_ACCEPTANCE','Created versioned common physical evaluator; preserved all old files; proceeding to saved candidate geometry repair',
        [p,STAGE/'PROVENANCE_AUDIT.md'],'SMOKE3_REQUALIFICATION',
        '0.179rad의 물리 한계 유래는 발견되지 않아 내부 솔버 스텝 제한으로 분류했습니다. A/B 공통 최종 평가는 기존 관절별 속도·가속도, 적응형 분기, 관절 한계 및 상세 충돌 기준을 유지합니다. 저장 후보 재검증과 공통 충돌 복구를 계속합니다.')
    atomic_text(OUT/'CURRENT_STATUS.md','# Current state\n\nIN_PROGRESS: common physical temporal acceptance unified after provenance audit and user authorization. Old condition7 resolved. No0.179rad aggregate trajectory rejection. Per-joint/collision/target/timing constraints unchanged. See master_autonomous/physical_acceptance_v4/.\n')

def run(case):
    verified_oracle_contract();g,c,natural=model();s=CommonPositionSolver(g,c,read(QUAL),natural);g.assign(natural);bounds=orbit_enclosures(g)
    target,hands,ts,source=load_input(case,g,natural);dt=float(np.median(np.diff(ts)));folder=V4/case
    if (folder/'SOURCE_POSITION_PASS.json').exists():print('REUSE_V4_PASS',case,flush=True);return
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m']
    qp,rp,record,passed=choose_candidate(case)
    candidate_folder=RUN/case/'aggregate_step_semantics_diagnostic_v1'
    # Residual/validity selects candidate provenance, never representation identity.
    if not passed and (candidate_folder/'DETAILED_GEOMETRY_0.json').exists():
        rp=candidate_folder/'DETAILED_GEOMETRY_0.json';record=read(rp);qp=Path(record['trajectory']['path'])
    q=np.load(qp)['q'].copy();initial=q.copy()
    atomic_json(folder/'INPUT.json',dict(source=file_record(qp),report=file_record(rp),acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json')))
    ap=RUN/case/'certified_anchor_recovery/CERTIFIED_ANCHORS.npz'
    if ap.exists():z=np.load(ap);anchors=z['q'].copy();mask=z['fixed_mask'].copy()
    else:anchors=q.copy();mask=np.zeros_like(q,dtype=bool)
    padding=read(ROOT/'configs/common_g1_morphology_adapter_v1.json')['collision_temporal_projection']['window_padding_candidates_frames'][:3]
    discovered={}
    for attempt in range(4):
        previous=folder/f'REPAIRED_{attempt-1}.npz'
        if attempt and previous.exists():q=np.load(previous)['q'].copy()
        met,actual,res,lb,cert,allow=qualify_numeric(s,q,target,hands,ts,bounds,slack)
        gp=folder/f'GEOMETRY_{attempt}.json'
        if gp.exists():geom=read(gp)['geometry']
        else:
            # A saved complete check is reusable only for the exact source q.
            cache=record.get('geometry',record.get('verification')) if passed and attempt==0 else None
            if cache:geom=read(Path(cache['path']))['geometry']
            elif attempt==0 and 'geometry' in record and isinstance(record['geometry'],list):geom=record['geometry']
            else:geom=[dict(frame=f,records=c.inspect(v,*h)) for f,(v,h) in enumerate(zip(q,hands))]
            atomic_json(gp,dict(geometry=geom,q_source=file_record(qp if attempt==0 else previous)))
        blocked=[]
        for r in geom:
            bad=[x for x in r['records'] if x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY')]
            if bad:blocked.append(r['frame'])
            for x in bad:discovered.setdefault(r['frame'],set()).add(tuple(x['geom_pair']))
        atomic_json(folder/f'CHECK_{attempt}.json',dict(metrics=met,blocked_frames=blocked,geometry=file_record(gp)))
        print('V4_CHECK',case,attempt,'numeric',met['pass_numeric'],'blocked',len(blocked),flush=True)
        if met['pass_numeric'] and not blocked:
            out=folder/'QUALIFIED_SOURCE_Q.npz'
            atomic_npz(out,q=q,EXECUTABLE_Q=q,RAW_REPRESENTATION_TARGET=target,EXECUTABLE_FK_POSITION=actual,
                POSITION_CORRECTION_MM=1000*res,CERTIFIED_LOWER_BOUND_MM=1000*lb,SOLVER_OPTIMALITY_GAP_MM=1000*(res-lb),
                CERTIFIED_UNREACHABLE=cert,common_hand_q=hands,source_timestamp=ts)
            atomic_json(folder/'SOURCE_POSITION_PASS.json',dict(metrics=met,trajectory=file_record(out),geometry=file_record(gp),
                source=source,acceptance=file_record(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json'),preparation_join='RECHECK_BEFORE_FREEZE'))
            return
        if attempt==3:break
        pp={};halo=padding[attempt]
        for f,pairs in discovered.items():
            for j in range(max(0,f-halo),min(len(q),f+halo+1)):pp.setdefault(j,set()).update(pairs)
        pp={f:sorted(v) for f,v in pp.items()}
        out=folder/f'REPAIRED_{attempt}.npz';mp=folder/f'REPAIRED_{attempt}.json'
        if not out.exists():
            start=time.monotonic();print('V4_REPAIR',case,attempt,'halo',halo,flush=True)
            q,fit=restore(s,target,hands,q,dt,allow,collision_pairs=pp,max_nfev=400,fixed_mask=mask,fixed_values=anchors)
            atomic_npz(out,q=q,RAW_REPRESENTATION_TARGET=target,common_hand_q=hands,source_timestamp=ts)
            atomic_json(mp,dict(fit=fit,runtime_s=time.monotonic()-start,collision_pairs={str(f):v for f,v in pp.items()}))
    atomic_json(folder/'BOUNDED_REPAIR_RESULT.json',dict(metrics=met,blocked_frames=blocked,next='DIAGNOSE_AND_COMMON_RETRY',global_infeasibility_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args()
    if a.case=='initialize':initialize()
    else:run(a.case)
