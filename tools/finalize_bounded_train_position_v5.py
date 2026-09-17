#!/usr/bin/env python3
"""Close only an exhausted position-qualification run, never a running search.

This is a scientific gate failure, not a claim of mathematical impossibility.
If all trajectories qualify, this script refuses to stop the master pipeline.
"""
from pathlib import Path
import sys,os,subprocess,datetime,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_train11_position_v5 import *
from tools.publish_train11_diagnostics_v5 import main as diagnostics
from tools.publish_train_raw_bound_diagnostics_v5 import main as raw_diagnostics
from tools.verify_train_position_candidate_parity_v5 import main as parity
from tools.publish_remaining_train_failures_v5 import main as remaining_failures

def main():
    verified_oracle_contract()
    active=[]
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]:
        parts=line.split();pid=int(parts[0])
        if pid==os.getpid():continue
        names=[Path(x).name for x in parts[1:]]
        if any(x.startswith(('run_train_','run_common_train_seed_recovery','orchestrate_remaining_train11')) and x.endswith('.py') for x in names):active.append(line.strip())
    if active:raise RuntimeError('Bounded searches still active; no terminal state written: '+repr(active))
    latest=max(p.stat().st_mtime for p in TRAIN.rglob('*') if p.is_file())
    if time.time()-latest<30:raise RuntimeError('Require at least30 seconds of stable completed TRAIN artifacts before terminal publication')
    parity();raw_diagnostics();remaining_failures();diagnostics()
    folder=OUT/'07_paper_artifacts/diagnostic_position_v5';diag=read(folder/'TRAIN11_POSITION_DIAGNOSTICS.json')
    counts=diag['counts']
    if all(v==11 for v in counts.values()):raise RuntimeError('All TRAIN11 trajectories qualify: continue to full 6D, do not stop.')
    retry=TRAIN/'INTERACTION_EP044/common_local_conic_v2/FINAL_CANDIDATE_global_position_audit_v1/common_global_box_retry_v1/COMPLETE.json'
    assert retry.exists(),'Additional common unresolved-frame search must finish first'
    failed=[r['case'] for r in diag['rows'] if not r['qualified']]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=ST5/'previous_status'/('bounded_completion_'+stamp)
    for p in [OUT/'CURRENT_STATUS.md',OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md',OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md',
              MASTER/'CURRENT_STAGE.md',MASTER/'LATEST_BLOCKER.md',MASTER/'CHATGPT_UPDATE.md',MASTER/'CHECKPOINT_STATE.json']:
        if p.exists():atomic_text(archive/p.relative_to(OUT),p.read_text())
    inventory=[]
    for r in diag['rows']:
        base=TRAIN/r['case']
        inventory.append(dict(case=r['case'],qualified=r['qualified'],
            completed_bounded_results=[file_record(p) for p in sorted(base.rglob('*RESULT.json')) if 'FIT' not in p.name],
            search_contracts=[file_record(p) for p in sorted(base.rglob('CONTRACT.json'))],
            numerical_fit_count=len(list(base.rglob('*FIT*.json')))))
    final=ST5/'bounded_train_completion';final.mkdir(parents=True,exist_ok=True)
    atomic_json(final/'BOUNDED_SEARCH_INVENTORY.json',dict(rows=inventory,active_searches=[],no_global_impossibility_claim=True))
    raw=read(folder/'RAW_CERTIFIED_MORPHOLOGY_DIAGNOSTICS.json')['aggregate']
    body=f'''# Bounded common position qualification — scientific blocker

## Scientific variable and provenance

The only intended method difference remains A=WRIST-TRAJECTORY-CENTRIC RAW TARGET GENERATION versus B=INTERACTION-CENTRIC RAW TARGET GENERATION. The source event clock, task/workspace registration, morphology handling, numerical search rules, hard limits, geometry classifier and temporal acceptance are common and method-blind. Case labels in runners locate inputs/outputs; they do not enter the solver objective. Raw targets are stored separately from executable q and correction diagnostics. Accepted-candidate source/state/timestamp/hash checks pass; overall dataset/training/physical parity is not claimed before those gates run.

Earlier physical experiments and the invalid standardized-grasp rebase remain diagnostic provenance only. No old physical result is mixed with the present rebuild. No real G1 hardware commands were issued. Existing raw-target competence and registration/event audits are retained; this work does not revise their spatial or temporal conventions.

## Decision

TRUE STOP CONDITION 4: the completed, bounded common realization portfolio has not constructed a complete physically valid executable trajectory for every predeclared TRAIN11 case. No candidate is promoted by dropping a physical, geometry, raw-fidelity or certified-optimality gate.

SMOKE3 is complete: A3/3 and B3/3. Fixed TRAIN11 is A{counts['WRIST']}/11 and B{counts['INTERACTION']}/11. Remaining cases: {', '.join(failed)}.

This is an operational failure to construct qualified trajectories under the declared finite searches, NOT a theorem that these trajectories or all raw targets are globally impossible. In particular, A32 has geometry-valid isolated witnesses but no qualified continuous realization; B44 samples201–202 remain bounded-search-no-witness, not certified-unreachable. Silently allowing a projection there would violate the contract.

## A24 is resolved

The original left_shoulder_roll_link / torso_link hard collisions at frames166–167 are removed. Detailed separations are0.023694569mm and0.023379672mm; proxy overlap remains diagnostic only. The left Cartesian residuals are9.998999261mm and9.999002410mm. The repaired window changes q at source frames120–213; raw targets and source timestamps are unchanged. The complete source trajectory and unchanged0.7s natural-start prefix pass geometry, limits, finite-state, adaptive-branch and per-joint velocity/acceleration checks.

The target-blind coupled-chain enclosure tightening of6.303063micrometres applies to both arms and methods. It is a more accurate model-only certificate, not a relaxed raw or collision threshold. The old0.179rad aggregate restriction remains internal-only. The10mm raw fidelity gate,10micrometre geometry tolerance,10micrometre certified-frame numerical slack,4.5rad/s and130rad/s² physical limits remain unchanged.

## Bounded alternatives executed

Forward, backward and independent seeding; common TRAIN-only seed banks; local sparse constrained restoration; conic trust-region and sum-slack searches; deterministic branch continuation; forward/backward stitching; terminal-wrist position-nullspace ranking; detailed-CAD boundary repair; analytic redundant branches and future-anchor bridges; alternative first-task branches under the same natural q0 and0.7s prefix; additional66-seed frame-independent audits; and an independent bounded global-box retry for unresolved B44 samples. Individual contracts, finite budgets, inputs, outputs and failures are retained in BOUNDED_SEARCH_INVENTORY.json. Ordinary ambiguous-containment-ray exceptions were handled fail-closed and interrupted searches resumed; they were not used as scientific stop reasons.

The extra B44 audit found valid isolated witnesses at frames194 and209. At201 and202, the best multi-start residuals were20.349904mm and20.156826mm. A separate differential-evolution box search and polish did not find a valid witness either. These are achieved search residuals, NOT certified minimum required corrections. No geometry threshold, target, source event, or method-specific exception was changed.

One exploratory alternate-prefix search initially used the solver posture prior instead of the authoritative natural q0. It was not used to promote an accepted trajectory. A new bounded version corrected this distinction and reran the same0.7s prefix with the authoritative q0. The independent final qualifier always used the authoritative q0. Two owned slow numerical searches were checkpointed and resumed with bounded early-rejection witness sampling; this never provides acceptance and never replaces the complete final collision classifier. These infrastructure corrections and all prior versions remain provenance.

## Valid raw morphology evidence

Across the same7588 TRAIN frames per method, the model-only outer enclosure certifies raw error greater than10mm for A{raw['WRIST']['certified_beyond_10mm_frames']}/7588 and B{raw['INTERACTION']['certified_beyond_10mm_frames']}/7588. Non-certified frames are not automatically reachable. These are raw error lower bounds, not achieved corrections or physical task-success rates.

## Downstream gates

Full6D solving, loaded Dex3 qualification, complete reference qualification, common-execution freeze, exact corrected-vs-old action diff, regeneration/retraining decision, paired dataset generation, ACT training, policy sanity, DEV35 smoke/freeze, all70 physical rollouts, stage success, TSR, confidence intervals, McNemar statistics and actual PhysX replay rendering are NOT RUN in this final rebuild. Orientation-interface audit alone passed; that is not6D qualification. No final freeze or completed final physical result is claimed.

## Next technically valid action

Continue common position feasibility research from the saved witnesses and candidate trajectories: establish a tighter model/joint-limit certificate for unresolved inner-workspace targets or construct a valid witness, and find a continuous physically admissible branch connecting the reachable/closest-feasible anchors. Then rerun whole-trajectory geometry and fixed-prefix qualification for all affected cases with identical rules. Only after both fixed TRAIN11 sets qualify may the existing master sequence proceed. No method-specific rescue, outcome-tuned projection, altered timing or relaxed physical bound is authorized or recommended.

## Paper-safe interpretation

The common local collision repair succeeds on SMOKE3 without changing representations or physical acceptance rules. The broader TRAIN qualification remains incomplete. The raw morphology certificates are a valid diagnostic representation result; no learned-policy or physical-task advantage is established. DEV35 remains repeatedly inspected development data. No untouched final-test, real-G1 success, target-domain visual deployment, or B superiority claim is supported.
'''
    bp=MASTER/'BLOCKERS/TRAIN11_BOUNDED_COMMON_POSITION_V5.md';atomic_text(bp,body)
    atomic_text(MASTER/'LATEST_BLOCKER.md',body)
    report=body.replace('# Bounded common position qualification — scientific blocker','# Final single-variable A/B rebuild and evaluation report — blocked before execution freeze',1)
    report+='\n## Exact candidate-level remaining failures\n\n'+(folder/'REMAINING_TRAIN_POSITION_FAILURES.md').read_text()
    report+='\n## Artifacts\n\n'
    evidence=[ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json',ST5/'smoke_diagnostics/A24_LOCAL_REPAIR_DETAIL.json',
        ST5/'full_chain_certificate/MODEL_ONLY_CERTIFICATE.json',ST5/'full_chain_certificate/VALIDATION.json',
        ST5/'orientation_interface_audit/ORIENTATION_CHAIN_AUDIT.json',ST5/'POSITION_CANDIDATE_PARITY_SNAPSHOT.json',folder/'TABLE_TRAIN11_POSITION_DIAGNOSTICS.md',
        folder/'TABLE_TRAIN11_POSITION_DIAGNOSTICS.csv',folder/'FIGURE_TRAIN11_POSITION_DIAGNOSTICS.png',
        folder/'FIGURE_TRAIN11_POSITION_DIAGNOSTICS.pdf',folder/'FIGURE_TRAIN11_POSITION_DIAGNOSTICS.svg',
        folder/'TABLE_RAW_CERTIFIED_MORPHOLOGY_DIAGNOSTICS.md',folder/'RAW_CERTIFIED_MORPHOLOGY_DIAGNOSTICS.json',folder/'FIGURE_RAW_CERTIFIED_LOWER_BOUNDS.png',
        folder/'REMAINING_TRAIN_POSITION_FAILURES.md',folder/'REMAINING_TRAIN_POSITION_FAILURES.json',folder/'REMAINING_TRAIN_POSITION_FAILURES.csv',
        final/'BOUNDED_SEARCH_INVENTORY.json',retry,bp]
    for p in evidence:
        assert p.exists();report+=f'- `{p}` — SHA256 `{file_record(p)["sha256"]}`\n'
    report+='\nFinal DEV35 paper result tables, stage-success figures and measured PhysX videos are absent because no final physical rollout was authorized by the qualification gates. They are not replaced by command-only animations or invented zero-success statistics. Earlier physical results remain PRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY. All prior runs and reports are preserved.\n'
    rp=OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md';atomic_text(rp,report)
    atomic_text(OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md','# Untouched final-test readiness\n\nREADY_FOR_UNTOUCHED_FINAL_TEST_COLLECTION = NOT_YET_QUALIFIED\n\nDo not present the existing DEV35 as untouched data. First resolve common TRAIN position qualification, then complete6D, loaded Dex3, reference qualification and frozen execution. Perform the exact supervision diff; regenerate/retrain both methods only when required, with a predeclared shared checkpoint rule. Freeze datasets, checkpoints, environments, scorer and runner after structural smoke. A genuinely fresh collection may then be registered before one-pass evaluation without further tuning. This is a prospective procedure, not an already frozen executable final-test protocol.\n')
    atomic_text(OUT/'CURRENT_STATUS.md',f'# Final single-variable rebuild\n\nStatus: MASTER_SINGLE_VARIABLE_AB_TRUE_SCIENTIFIC_BLOCKER\n\nA24 local repair and SMOKE3 pass: A3/3, B3/3. Fixed TRAIN11: A{counts["WRIST"]}/11, B{counts["INTERACTION"]}/11. No completed common trajectory for every case after bounded shared recovery. No global impossibility claim.\n\nReport: `{rp}`\n\nNext required gate: complete common TRAIN11 executable-position qualification, then full6D. All final physical/training stages remain NOT_RUN.\n')
    korean=f"A24 충돌 복구와 SMOKE3 A3/3·B3/3은 통과했습니다. 그러나 공통 유한 탐색을 마친 고정 TRAIN11은 A{counts['WRIST']}/11·B{counts['INTERACTION']}/11로 전체 물리 유효 궤적을 구성하지 못했습니다. 이는 전역 불가능성 증명이 아닙니다. 특히 B44의 미해결 목표를 임의로 투영하지 않았고 A/B 목표·시간·물리 한계·충돌 기준은 유지했습니다. TRUE STOP4에 따라 6D·Dex3·학습·DEV35 실행을 시작하지 않았습니다. 진단 표·그림·해시·차단 보고서를 저장했으며 CLI는 과학적 게이트에서 중단합니다."
    log('FIXED_TRAIN11_POSITION','TRUE_SCIENTIFIC_BLOCKER',[ST5/'INDEPENDENT_SMOKE_AND_PREPARATION_RESULT.json',folder/'TRAIN11_POSITION_DIAGNOSTICS.json',retry],
        'Complete common physical trajectories not constructed for all fixed TRAIN11 cases after bounded shared searches',
        'BOUNDED_COMMON_POSITION_REALIZATION_NOT_QUALIFIED',
        'Preserved all evidence; did not bypass raw/geometry/temporal gates or run downstream stages',
        [rp,bp,final/'BOUNDED_SEARCH_INVENTORY.json',folder/'DIAGNOSTIC_ARTIFACT_MANIFEST.json'],
        'COMPLETE_COMMON_TRAIN11_POSITION_BEFORE_FULL_6D',korean)
    atomic_json(final/'FINAL_BLOCKER_MANIFEST.json',dict(status='MASTER_SINGLE_VARIABLE_AB_TRUE_SCIENTIFIC_BLOCKER',
        true_stop_condition=4,global_impossibility_proven=False,counts=counts,remaining=failed,active_searches=[],
        final_pipeline_results_ready=False,required_success_artifact_availability={str(p.relative_to(OUT)):p.exists() for p in [
            OUT/'03_common_execution_freeze/COMMON_EXECUTION_FREEZE_MANIFEST.json',
            OUT/'07_paper_artifacts/TABLE_SINGLE_VARIABLE_AB_DEV35_RESULTS.md',OUT/'07_paper_artifacts/TABLE_SINGLE_VARIABLE_AB_DEV35_RESULTS.csv',
            OUT/'07_paper_artifacts/FigXX_Single_Variable_AB_DEV35_double.png',OUT/'07_paper_artifacts/FigXX_Single_Variable_AB_DEV35_double.pdf',OUT/'07_paper_artifacts/FigXX_Single_Variable_AB_DEV35_double.svg',
            OUT/'A_DEV35_FINAL_TOP_35SPLIT.mp4',OUT/'A_DEV35_FINAL_OVERVIEW_35SPLIT.mp4',OUT/'B_DEV35_FINAL_TOP_35SPLIT.mp4',OUT/'B_DEV35_FINAL_OVERVIEW_35SPLIT.mp4']},
        implementation_hashes=[file_record(p) for p in sorted((ROOT/'tools').glob('*v5.py')) if p.name.startswith(('run_train_','publish_train','finalize_bounded','verify_train','common_full_chain'))]+[file_record(ROOT/'tools'/n) for n in ['common_fast_hard_witness_v6.py','common_training_seed_bank_v9.py','run_common_train_seed_recovery_v8.py','run_train_prepared_branch_v6.py','common_geometry_confirmed_collision.py']],
        artifacts=[file_record(p) for p in [*evidence,rp,OUT/'CURRENT_STATUS.md',MASTER/'CHATGPT_UPDATE.md',OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md']]))
    print(f'''==================================================
MASTER ALOHA→G1 SINGLE-VARIABLE A/B RESULTS
==================================================
STARTUP PREPARATION
Natural q0 preserved: YES
Common preparation duration: 0.7 s / 21 prefix frames
Preparation convention identical A/B: YES
--------------------------------------------------
RAW CARTESIAN FEASIBILITY — FIXED TRAIN11
A certified beyond10mm: {raw['WRIST']['certified_beyond_10mm_frames']}/7588
B certified beyond10mm: {raw['INTERACTION']['certified_beyond_10mm_frames']}/7588
All-frame reachable fractions: NOT ESTABLISHED
All-frame achieved correction distributions: NOT QUALIFIED
--------------------------------------------------
EXECUTABLE POSITION
SMOKE3: A3/3, B3/3
Fixed TRAIN11: A{counts['WRIST']}/11, B{counts['INTERACTION']}/11
Remaining unqualified: {', '.join(failed)}
Qualified trajectories: hard collision0, hard-limit violations0, branch discontinuities0
Accepted candidates: {sum(counts.values())}/22; this is not a complete position freeze.
--------------------------------------------------
FULL6D: NOT RUN
LOADED DEX3: NOT RUN; no14/14 result claimed
EXACT ACTION DIFF: NOT RUN
DATASETS REGENERATED: NO
ACT RETRAINING REQUIRED: UNDETERMINED pending exact diff
ACT RETRAINED: NO
ACT-A / ACT-B corrected checkpoints: NOT AVAILABLE
--------------------------------------------------
DEV35
A/B valid physical runs: NOT RUN /35 each
A/B approach, grasp, lift, handoff, ownership, transport, bin, full task: NOT RUN
B-A and McNemar: NOT COMPUTABLE
--------------------------------------------------
DIAGNOSTIC FIGURE: {folder/'FIGURE_TRAIN11_POSITION_DIAGNOSTICS.png'}
DIAGNOSTIC TABLE: {folder/'TABLE_TRAIN11_POSITION_DIAGNOSTICS.md'}
FINAL DEV35 MAIN FIGURE / TABLE: NOT GENERATED (no physical results)
A/B TOP / OVERVIEW PHYSX VIDEOS: NOT GENERATED (no measured physical traces)
FINAL REPORT: {rp}
CHATGPT UPDATE: {MASTER/'CHATGPT_UPDATE.md'}
TRUE STOP CONDITION:4 — bounded common position realization not qualified
Global physical impossibility: NOT PROVEN
==================================================''',flush=True)
    print('MASTER_SINGLE_VARIABLE_AB_TRUE_SCIENTIFIC_BLOCKER',flush=True)

if __name__=='__main__':main()
