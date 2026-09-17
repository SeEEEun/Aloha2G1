#!/usr/bin/env python3
"""Queue-safe ingestion and necessary-condition audit for the master contract.

Never treats an old 95% trajectory pass as an every-frame execution pass.
This diagnostic does not alter the solver, initial state, clocks or targets.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import time
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.cartesian_reachability_forensic import DEST,BASELINE,BASELINE_CONTRACT,model
from tools.common_framewise_reachability_oracle import FramewiseReachabilityOracle
from tools.common_g1_position_bounds import outer_enclosures,residual_lower_bounds
from tools.final_single_variable_prepare import OUT,read,file_record
from tools.run_reference_motion_scientific_reset import atomic_json,atomic_text,atomic_csv

MASTER=OUT/'master_continuation'
INITIAL=ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json'
QUAL=OUT/'01_registration/POSITION_SOLVER_QUALIFICATION_CONTRACT.json'
PROCESS_NAMES=('run_common_position_anchor_recovery.py','densify_cartesian_reachability.py',
               'finalize_cartesian_reachability_forensic.py','framewise_reachability_oracle.py')


def now():return datetime.now(timezone.utc).isoformat()


def active_predecessors():
    rows=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            argv=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        args=[a.decode(errors='replace') for a in argv if a]
        # Exact script basenames avoid matching a shell/monitor quoted command.
        if any(Path(a).name in PROCESS_NAMES for a in args):rows.append({'pid':int(path.name),'argv':args})
    return rows


def fingerprints(paths):return [file_record(p) for p in paths]


def verified_oracle_contract():
    """Preserve execution hashes; permit ONLY the requested report publication.

    The predecessor accidentally included its report destination among frozen
    dependencies. Its original exact bytes remain verified in the archive.
    No model, solver, target, timing, limit or geometry hash is substituted.
    """
    baseline=read(BASELINE_CONTRACT);oracle=read(DEST/'ORACLE_CONTRACT.json');substitutions=[]
    records=baseline['protected_artifacts']+baseline['new_implementations']+oracle['protected']+oracle['implementations']
    for record in records:
        path=Path(record['path']);actual=file_record(path)
        if actual['sha256']==record['sha256']:continue
        assert path==OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md',str(path)
        assert 'MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER' in path.read_text()
        archived=MASTER/'PREVIOUS_FINAL_REPORT.md';archive_record=file_record(archived)
        assert archive_record['sha256']==record['sha256'],'Original report archive does not match the frozen report'
        substitutions.append({'original_frozen_record':record,'verified_original_archive':archive_record,
                              'master_report_publication':actual,'execution_dependency_changed':False})
    return oracle,substitutions


def log(stage,status,inputs,problem,root_cause,action,artifacts,next_stage,korean=None):
    path=MASTER/'MASTER_RUN_LOG.jsonl'
    row={'timestamp':now(),'stage':stage,'status':status,'authoritative_inputs':fingerprints(inputs),
         'observed_problem':problem,'root_cause_classification':root_cause,'action_taken':action,
         'artifacts_created':fingerprints(artifacts),'next_stage':next_stage}
    previous=path.read_text() if path.exists() else ''
    for line in previous.splitlines():json.loads(line)
    atomic_text(path,previous+json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    atomic_text(MASTER/'CURRENT_STAGE.md',f'# Master continuation\n\nTimestamp: {row["timestamp"]}\n\nStage: {stage}\n\nStatus: {status}\n\nRoot cause: {root_cause}\n\nObserved problem: {problem or "none"}\n\nAction: {action}\n\nNext stage: {next_stage}\n')
    if korean:
        atomic_text(MASTER/'CHATGPT_UPDATE.md',korean+'\n');print(korean,flush=True)


def queue_handoff(interval=30):
    files=[OUT/'CURRENT_STATUS.md',OUT/'CURRENT_STATUS.json',DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.md',
           DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json',DEST/'FINAL_HASH_MANIFEST.json']
    # No filesystem writes until no predecessor is active and one full interval
    # has elapsed with identical persisted status/report contents.
    while True:
        if active_predecessors():time.sleep(interval);continue
        first=fingerprints(files);started=now();time.sleep(interval)
        if not active_predecessors() and first==fingerprints(files):break
    original_status=(OUT/'CURRENT_STATUS.md').read_text()
    report=read(DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json')
    manifest=read(DEST/'FINAL_HASH_MANIFEST.json')
    records=manifest['artifacts']+manifest['dense_files']+manifest['recovery_files']
    for r in records:assert file_record(Path(r['path']))==r,r['path']
    _,substitutions=verified_oracle_contract()
    atomic_text(MASTER/'PRECEDING_CURRENT_STATUS.md',original_status)
    atomic_json(MASTER/'QUEUE_HANDOFF.json',{'started':started,'completed':now(),'verification_interval_s':interval,
        'active_predecessors':[],'stable_artifacts':first,'verified_artifact_count':len(records),
        'preceding_dense_forensic_complete':report['independent_validation']['oracle_frames']==602,
        'preceding_recovery_smoke_complete':report['recovery_smoke_complete']})
    log('QUEUE_SAFE_HANDOFF','PASS',files,'','MIXED_SOLVER_AND_TARGET_FEASIBILITY',
        'Waited for a second stable 30-second interval; verified all predecessor hashes; no dense rerun or process interruption.',
        [MASTER/'QUEUE_HANDOFF.json',MASTER/'PRECEDING_CURRENT_STATUS.md'],'FORENSIC_DECISION')
    if substitutions:
        mapping=MASTER/'FROZEN_REPORT_PROVENANCE_MAP.json'
        atomic_json(mapping,{'report_publication_only':True,'execution_hash_substitutions':0,'records':substitutions})
        infrastructure=MASTER/'BLOCKERS/REPORT_PUBLICATION_PROVENANCE.md'
        atomic_text(infrastructure,'# Report publication provenance conflict — RESOLVED\n\nThe old geometry freeze hashed the legacy top-level final report as well as executable dependencies. The master explicitly requests publishing a new report at that path. The original report bytes are preserved in PREVIOUS_FINAL_REPORT.md and match their old frozen SHA256. Only this reporting path has a provenance mapping. Every model, solver, target, event clock, geometry rule and limit remains checked at its original path and hash. The old manifests and implementations were not edited. This is not a scientific acceptance waiver.\n')
        log('REPORT_PUBLICATION_PROVENANCE','RESOLVED',[BASELINE_CONTRACT,MASTER/'PREVIOUS_FINAL_REPORT.md'],
            'Old dependency verifier included the user-requested report output path.','REPORTING_PROVENANCE_ONLY',
            'Archived exact prior report; verified its old hash; explicitly mapped reporting supersession only.',[mapping,infrastructure],
            'FORENSIC_DECISION','보고서 출력 경로와 과거 동결 해시의 충돌을 해결했습니다. 원본 보고서는 원래 SHA256 그대로 보관·검증하며 실행 의존성의 예외는 없습니다. A/B 공정성에는 영향이 없고 CLI는 경계 조건 검증을 재개합니다.')
    assert report['root_cause']=='MIXED'
    decision={'classification':'MIXED_SOLVER_AND_TARGET_FEASIBILITY','source':file_record(DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json'),
        'dense_cases':report['dense_cases'],'method_gap_statistics':report['method_gap_statistics'],
        'old_recovery_complete':report['recovery_smoke_complete'],
        'new_contract_interpretation':'Certified morphology gaps are allowed to receive transparent common closest-feasible realization. The old 95% raw gate is not the new executable gate.'}
    atomic_json(MASTER/'CARTESIAN_FORENSIC_DECISION.json',decision)
    atomic_text(MASTER/'CARTESIAN_FORENSIC_DECISION.md','# Cartesian forensic decision\n\nMIXED_SOLVER_AND_TARGET_FEASIBILITY\n\nThe dense forensic is complete; the preceding recovery smoke is incomplete and unqualified. All 602 old failures were inspected: A has 161 reachable witnesses, 290 certified-unreachable frames and four uncertified no-witness frames; B49 has 147 reachable witnesses and no certified-unreachable frames.\n\nThe new master permits explicit closest-feasible realization of certified-unreachable targets. Consequently the old A 0/3 raw-fidelity result is NOT, by itself, a reason to stop this continuation. Raw and executable metrics must stay separate. A new every-frame necessary-condition check is required.\n')
    log('FORENSIC_DECISION','PASS',[DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json'],'Prior sequential candidate did not qualify.',
        decision['classification'],'Accepted the new dual raw/executable distinction; did not reuse the old 95% gate.',
        [MASTER/'CARTESIAN_FORENSIC_DECISION.json',MASTER/'CARTESIAN_FORENSIC_DECISION.md'],'INITIAL_BOUNDARY_NECESSARY_CONDITION',
        '선행 포렌식 602프레임과 해시 검증을 완료했습니다. 원인은 솔버 실패와 타깃 도달 불가능성이 혼재합니다. A/B 공정성은 유지됩니다. 새 이중 판정 규칙에 따라 고정 초기상태와 프레임별 조건의 양립 가능성을 공통 방식으로 확인하며 CLI는 계속 진행합니다.')
    return report


def boundary_audit(report):
    cfg=verified_oracle_contract()[0]['oracle'];g1,collision,natural=model();oracle=FramewiseReachabilityOracle(g1,collision,cfg,natural)
    g1.assign(natural);balls=outer_enclosures(g1)
    initial=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad'])
    rows=[];proofs=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in (0,24,49):
            name=f'{mode}_EP{ep:03d}';source=BASELINE/(name+'.npz')
            with np.load(source,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['initial_q'],initial)
                np.testing.assert_array_equal(z['q'][0],initial)
                target=z['RAW_REPRESENTATION_TARGET'][0].copy();hands=z['common_hand_q'][0].copy()
                timestamp=float(z['source_timestamp'][0]);g1.assign(initial,*hands)
                fk=np.array([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
                error=np.linalg.norm(fk-target,axis=1)
                np.testing.assert_allclose(error,z['position_residual_m'][0],rtol=0,atol=1e-12)
            records=collision.inspect(initial,*hands)
            existing=DEST/'dense'/f'{name}_F0000.json'
            witness=read(existing) if existing.exists() else oracle.solve(target,hands)
            assert witness['classification']=='FRAME_REACHABLE'
            best=witness['best'];wq=np.asarray(best['q']);g1.assign(wq,*hands)
            wfk=np.array([g1.data.xpos[g1.wrist_ids[s]].copy() for s in ('left','right')])
            werr=np.linalg.norm(wfk-target,axis=1)
            wr=collision.inspect(wq,*hands)
            assert werr.max()<=.01 and not any(r['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for r in wr)
            assert np.all(wq>=g1.arm_limits[:,0]) and np.all(wq<=g1.arm_limits[:,1])
            proof={'case':name,'frame':0,'source_timestamp':timestamp,'RAW_REPRESENTATION_TARGET':target.tolist(),
                'fixed_initial_q':initial.tolist(),'fixed_initial_fk':fk.tolist(),'fixed_initial_residual_mm':(1000*error).tolist(),
                'fixed_initial_geometry':records,'reachable_witness_q':wq.tolist(),'reachable_witness_residual_mm':(1000*werr).tolist(),
                'reachable_witness_geometry':wr,'witness_seed_id':best['seed_id'],
                'certified_lower_bound_mm':(1000*residual_lower_bounds(target,balls)).tolist(),
                'classification':'FRAME_REACHABLE','necessary_condition':'q[0] == fixed_initial_q AND max_wrist_residual(q[0]) <= 0.01 m',
                'necessary_condition_pass':bool(error.max()<=.01),'source':file_record(source),
                'witness_provenance':file_record(existing) if existing.exists() else {'kind':'NEW_BOUNDED_COMMON_FRAME_ZERO_ORACLE','oracle_contract':file_record(DEST/'ORACLE_CONTRACT.json')}}
            proofs.append(proof)
            rows.append({'case':name,'frame':0,'left_residual_mm':1000*error[0],'right_residual_mm':1000*error[1],
                'maximum_residual_mm':1000*error.max(),'reachable_witness_maximum_mm':1000*werr.max(),
                'new_every_frame_gate_pass':bool(error.max()<=.01)})
            print('BOUNDARY_AUDIT',rows[-1],flush=True)
    assert all(not r['necessary_condition_pass'] for r in proofs)
    result={'status':'INITIAL_BOUNDARY_AND_EVERY_FRAME_RAW_GATE_INCOMPATIBLE','proof_scope':'Fixed existing q[0], original source-indexed frame zero and the requested <=10mm reachable-frame rule. Not a claim that no trajectory exists after changing the startup convention.',
        'rows':proofs,'raw_targets_modified':False,'initial_state_modified':False,'clock_modified':False,
        'collision_tolerance_m':1e-5,'closest_feasible_slack_selected':False,
        'slack_note':'The <=2mm closest-feasible slack applies only to certified-unreachable targets; all six boundary targets have actual reachable witnesses, so no choice of that slack repairs this contradiction.',
        'upstream_forensic_root_cause':'MIXED_SOLVER_AND_TARGET_FEASIBILITY','script':file_record(Path(__file__)),
        'authoritative_initial_state':file_record(INITIAL),'original_execution_contract':file_record(QUAL)}
    atomic_json(MASTER/'INITIAL_BOUNDARY_PROOF.json',result)
    atomic_csv(MASTER/'INITIAL_BOUNDARY_TABLE.csv',rows)
    text='# Common initial-boundary diagnostic\n\nNOT DEV35 performance; offline TRAIN SMOKE3 necessary-condition audit.\n\n| Case | Left residual mm | Right residual mm | Reachable witness max mm | Every-frame gate |\n|---|---:|---:|---:|---|\n'
    for r in rows:text+=f"| {r['case']} | {r['left_residual_mm']:.6f} | {r['right_residual_mm']:.6f} | {r['reachable_witness_maximum_mm']:.9f} | FAIL |\n"
    atomic_text(MASTER/'INITIAL_BOUNDARY_TABLE.md',text)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.bar([r['case'].replace('WRIST_EP','A').replace('INTERACTION_EP','B') for r in rows],[r['maximum_residual_mm'] for r in rows],color=['#3975a8']*3+['#dd8b32']*3)
    ax.axhline(10,color='black',ls='--',label='Required reachable-frame gate: 10 mm')
    ax.set_ylabel('Frame-0 maximum wrist residual (mm)');ax.set_title('Fixed natural initial state vs. unchanged raw targets\nTRAIN diagnostic — not a DEV35 physical result');ax.legend()
    figures=[]
    for suffix in ('png','pdf','svg'):
        path=MASTER/f'INITIAL_BOUNDARY_DIAGNOSTIC.{suffix}';fig.savefig(path,dpi=180);figures.append(path)
    plt.close(fig)
    return result,figures


def finish(report,boundary,figures):
    blocker=MASTER/'BLOCKERS/INITIAL_BOUNDARY_CONTRACT_CONFLICT.md'
    text='# Scientific blocker: initial-boundary contract conflict\n\n'
    text+='The new master does allow common closest-feasible realization of certified-unreachable A targets. That is not the blocking condition here.\n\n'
    text+='At source frame zero, all six SMOKE3 trajectories use the same authoritative natural initial arm state. Its wrist FK is 56.18–56.45 mm from A targets and 67.40–67.48 mm from B targets. Every target has an independently rechecked collision-valid <=10mm witness. These are reachable targets, so the new master requires <=10mm raw residual, not closest-feasible substitution.\n\n'
    text+='For any trajectory with q(0)=q_initial, FK(q(0)) is fixed. Its measured residual exceeds 10mm. No reseeding, null-space continuation, optimality slack, later-frame correction or collision-rule change can satisfy both equalities at t=0. This necessary-condition failure precludes the complete requested executable trajectory under the current boundary convention. It is independent of A/B identity and of task success.\n\n'
    text+='## Common alternatives examined\n\n- Jump to the reachable witness at frame zero: changes the matched natural initial state and introduces an unaccounted initial transition. Not applied.\n- Project the frame-zero target: prohibited for these proven reachable targets. Not applied.\n- Exempt startup frames or retain the old 95% rule: contradicts the requested every-frame rule. Not applied.\n- Add a logged, identical-duration preparation interval before source time zero: potentially valid for a revised experiment, but requires explicitly defining whether it belongs to the natural-start task, its raw-target validity mask, event-clock mapping, image/state samples and scoring interval. It is not silently introduced as already-qualified source motion.\n- Redefine one shared source-derived natural initial state: potentially valid after a common boundary-contract amendment and requalification, not a per-method witness initialization. No evidence establishes an already-authorized replacement.\n\n'
    text+='This is stop condition 4 under the current constraints, not a global impossibility claim for all amended startup conventions. No A/B-specific handling, outcome tuning, corrupted source or real hardware is involved. A common startup-contract choice is required before proceeding. The fixed 11-episode TRAIN set and all later gates remain unrun because SMOKE3 already fails this necessary condition.\n\n'
    text+='No optimizer was run indefinitely; the completed dense forensic was reused. Only two missing frame-zero witnesses were obtained with the frozen common bounded oracle; all six witnesses and the fixed FK were checked independently. No executable closest-feasible trajectory has been qualified and no numerical slack has been selected to hide the failure.\n'
    atomic_text(blocker,text)
    korean='실패: 새 프레임별 위치 조건이 기존 자연 초기상태와 양립하지 않습니다. 원인: 도달 가능한 프레임 0 타깃에 대해 고정 초기 FK 오차가 A 약 56 mm, B 약 67 mm로 10 mm를 초과합니다. A/B 공정성은 유지되며 타깃·초기상태·충돌 기준은 바꾸지 않았습니다. 공통 IK 증인과 FK를 재검증했으나, 해결에는 공통 준비 구간 또는 초기상태/시간 경계 계약의 명시적 재정의가 필요합니다. CLI는 과학적 차단 조건에서 중단했으며 6D·Dex3·학습·DEV35 평가는 실행하지 않았습니다.'
    artifacts=[MASTER/'INITIAL_BOUNDARY_PROOF.json',MASTER/'INITIAL_BOUNDARY_TABLE.csv',MASTER/'INITIAL_BOUNDARY_TABLE.md',blocker,*figures]
    log('COMMON_EXECUTABLE_POSITION_PREFLIGHT','MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER',
        [INITIAL,QUAL,DEST/'FINAL_CARTESIAN_REACHABILITY_FORENSIC.json'],
        'Fixed natural frame-zero state violates the new raw <=10mm condition on six proven-reachable targets.',
        'INITIAL_BOUNDARY_CONTRACT_CONFLICT',
        'Verified exact fixed-state FK and collision-valid framewise witnesses. Evaluated common alternatives; no silent startup exemption, retiming or initialization change.',
        artifacts,'REQUIRES_COMMON_STARTUP_BOUNDARY_CONTRACT; ALL_DOWNSTREAM_GATES_BLOCKED',korean)
    final=OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md'
    if final.exists() and not (MASTER/'PREVIOUS_FINAL_REPORT.md').exists():atomic_text(MASTER/'PREVIOUS_FINAL_REPORT.md',final.read_text())
    sections=[('Cartesian forensic result','MIXED_SOLVER_AND_TARGET_FEASIBILITY; complete dense forensic reused, 634 hashes verified.'),
      ('Solver-recoverable versus certified-unreachable frames','A: 161 reachable among 455 old failing frames; 290 certified unreachable and four unresolved bounded-search frames. B49: 147/147 old failures reachable; zero certified unreachable. Scope is TRAIN smoke, not DEV35.'),
      ('Common morphology realization','Authorized in principle for certified-unreachable targets, with raw/executable outputs separated. No full executable trajectory qualified: the new every-frame gate contradicts the fixed initial boundary on reachable targets. No slack was selected and no projection was applied.'),
      ('Raw A/B feasibility','A: 1772/2066 frames have known raw <=10mm witnesses from baseline plus dense oracle; 290/2066 certified unreachable; 4/2066 unresolved. B49: 686/686 frames have baseline/oracle witnesses. This is framewise feasibility, not sequential success. A correction-to-10mm-gate lower/upper bounds over 294 no-witness frames: mean 46.700/47.377 mm; p95 104.820/105.426 mm; max 106.552/107.153 mm. These are diagnostic bounds, not applied executable corrections or exact global minima.'),
      ('Full 6D qualification','NOT RUN — position prerequisite blocked.'),('Loaded Dex3 qualification','NOT RUN — no new 14/14 or measured-limit pass claimed.'),
      ('Exact action-target diff','NOT RUN — no corrected qualified supervision exists.'),('Regeneration/retraining decision','NOT DETERMINED; datasets not regenerated and ACT not retrained.'),
      ('Dataset parity','NOT AUDITED for a corrected dataset. No dataset was changed.'),('ACT training parity','NO NEW TRAINING. Existing checkpoints were neither replaced nor selected by physical results.'),
      ('DEV35 physical results','NOT RUN. Valid rollouts A 0/35, B 0/35 executed in this continuation; task success counts and TSR are NOT AVAILABLE, not 0%.'),
      ('Stage success','NOT AVAILABLE.'),('Full task success rate','NOT AVAILABLE.'),('Matched statistics','NOT AVAILABLE; no fabricated confidence intervals or McNemar result.'),
      ('Figures and replay paths','Valid TRAIN diagnostic: master_continuation/INITIAL_BOUNDARY_DIAGNOSTIC.png/.pdf/.svg and INITIAL_BOUNDARY_TABLE.md. Dense forensic diagnostic remains under 02_common_execution_qualification/cartesian_reachability_forensic_v1/. Required DEV35 result figure/table and actual PhysX replay videos are NOT GENERATED.'),
      ('Limitations','The prior recovery smoke was incomplete and not promoted. This continuation establishes a fixed-boundary incompatibility, not universal impossibility after amending the startup convention. No new position/6D/Dex3 execution freeze exists. No real hardware was used.'),
      ('Paper-safe interpretation','Only a TRAIN-side morphology/solver diagnostic is supported. No DEV35 TSR, untouched test performance, real-G1 success, B superiority or autonomous target-domain deployment is claimed.')]
    body='# Final single-variable A/B continuation report\n\nMASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER\n\n'
    body+='Scientific stop: no complete trajectory can meet both the current fixed natural q(0) and the requested <=10mm gate on every proven-reachable source frame. The new closest-feasible rule for certified-unreachable targets does not apply at frame zero.\n\n'
    body+='See master_continuation/BLOCKERS/INITIAL_BOUNDARY_CONTRACT_CONFLICT.md and INITIAL_BOUNDARY_PROOF.json for the exact evidence and common startup alternatives.\n'
    for i,(title,content) in enumerate(sections,1):body+=f'\n## {i}. {title}\n\n{content}\n'
    atomic_text(final,body)
    ready=OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md'
    if ready.exists() and not (MASTER/'PREVIOUS_READY_FOR_UNTOUCHED_FINAL_TEST.md').exists():atomic_text(MASTER/'PREVIOUS_READY_FOR_UNTOUCHED_FINAL_TEST.md',ready.read_text())
    atomic_text(ready,'# Untouched final-test readiness\n\nREADY_FOR_UNTOUCHED_FINAL_TEST_EVALUATION = NO\n\nThe common position boundary is scientifically blocked; execution, datasets, checkpoints and scorer have not received the requested final freeze. DEV35 remains development data. No untouched manifest has been certified in this blocked continuation.\n\nAfter resolving the common startup contract: qualify position on SMOKE3 and the frozen TRAIN11, then 6D, loaded Dex3, reference pipeline, exact action audit, paired dataset/training parity, policy sanity and DEV smoke. Freeze and hash execution, checkpoints, scorer and membership before collecting a new independent FINAL_TEST manifest. Fresh episodes must not be used for calibration, debugging, checkpoint selection or visual tuning. Evaluate the frozen paired pipeline once, keep normal failures, rerun only infrastructure-invalid cases and preserve measured physics traces. Do not call current DEV35 results unseen test performance.\n')
    atomic_json(MASTER/'MASTER_RESULT.json',{'status':'MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER',
      'forensic_classification':'MIXED_SOLVER_AND_TARGET_FEASIBILITY','position_complete_executable_counts':{'WRIST':0,'INTERACTION':0},
      'position_scope':'new every-frame contract; necessary-condition failures, not six newly optimized trajectory rollouts',
      'stop_condition':4,'blocker':file_record(blocker),'boundary_proof':file_record(MASTER/'INITIAL_BOUNDARY_PROOF.json'),
      'full_6d':'NOT_RUN','loaded_dex3':'NOT_RUN','actions_changed':'NOT_AUDITED','datasets_regenerated':False,'ACT_retrained':False,
      'DEV35_rollouts_executed':{'WRIST':0,'INTERACTION':0},'DEV35_TSR':None,'physical_videos':[],
      'final_report':file_record(final),'fresh_test_readiness':file_record(ready),'source_targets_modified':False})
    _,substitutions=verified_oracle_contract()
    atomic_json(MASTER/'FROZEN_REPORT_PROVENANCE_MAP.json',{'report_publication_only':True,'execution_hash_substitutions':0,'records':substitutions})
    all_files=[p for p in MASTER.rglob('*') if p.is_file() and p.name!='MASTER_ARTIFACT_MANIFEST.json']+[final,ready]
    atomic_json(MASTER/'MASTER_ARTIFACT_MANIFEST.json',{'artifacts':fingerprints(sorted(all_files)),'code':file_record(Path(__file__))})
    print('MASTER_SINGLE_VARIABLE_AB_SCIENTIFIC_BLOCKER',flush=True)


if __name__=='__main__':
    report=queue_handoff();boundary,figures=boundary_audit(report);finish(report,boundary,figures)
