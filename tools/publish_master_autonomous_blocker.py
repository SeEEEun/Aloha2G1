#!/usr/bin/env python3
"""Publish only a completed bounded scientific stop, never fabricated DEV results."""
from pathlib import Path
import sys,json,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.master_autonomous_diagnostics import generate
from tools.run_master_autonomous import MASTER,RUN,QUAL,INITIAL,OUT,DEST,read,file_record,atomic_json,atomic_text,log
from tools.master_autonomous_provenance import verified_oracle_contract

TERMINAL='MASTER_SINGLE_VARIABLE_AB_TRUE_SCIENTIFIC_BLOCKER'


def run():
    # Publishing must not interrupt a running recovery or skip a new numeric pass.
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a).name.startswith('run_autonomous_') and Path(a).suffix=='.py' for a in argv):active.append(argv)
    assert not active,active
    for case in ('WRIST_EP024','WRIST_EP049'):
        folder=RUN/case/'local_dense_continuation_v1'
        assert not (folder/'NUMERIC_PASS.json').exists(),'A new numeric pass needs collision qualification, not a stop'
        assert (folder/'BOUNDED_SEARCH_COMPLETE.json').exists(),'Final bounded recovery is incomplete'
    oracle,substitutions=verified_oracle_contract()
    name='bounded_common_execution_v1';diag=MASTER/'diagnostics'/name
    if not (diag/'DIAGNOSTIC_MANIFEST.json').exists():generate(name)
    d=read(diag/'TRAIN_POSITION_DIAGNOSTIC.json');assert d['source_position_counts']=={'WRIST':1,'INTERACTION':3}
    arows=[r for r in d['rows'] if r['mode']=='WRIST'];brows=[r for r in d['rows'] if r['mode']=='INTERACTION']
    gap=read(diag/'CERTIFIED_MORPHOLOGY_GAP.json');bgap=read(diag/'B_CERTIFIED_MORPHOLOGY_GAP.json')
    windows=[]
    for case in ('WRIST_EP000','WRIST_EP024','WRIST_EP049'):
        for sub in ('hard_window_forensic','trust_window_forensic'):
            p=RUN/case/sub/'SUMMARY.json';assert p.exists();windows.extend(read(p)['rows'])
    budget=dict(hard_window_searches=len(windows),hard_window_relaxed_witnesses=sum(r['info']['relaxed_necessary_constraints_pass'] for r in windows),
        hard_window_global_impossibility_proofs=0,whole_trajectory_families=['propagated paths','independent witnesses','feasible history','unsmoothed anchors','fixed certified anchors','higher-accuracy sparse refinement','local dense refinement'],
        local_dense_window_padding_frames=[16,32,48],raw_targets_modified=False,scientific_gates_weakened=False)
    atomic_json(MASTER/'BOUNDED_RECOVERY_ACCOUNTING.json',budget)
    blocker=MASTER/'BLOCKERS/COMMON_TEMPORAL_REALIZATION_NOT_QUALIFIED.md'
    text='# Bounded common trajectory realization remains unqualified\n\n'+TERMINAL+'\n\n'
    text+='Stop condition 4, operational scope: the completed bounded common realization did not construct a complete accepted A/B SMOKE3 trajectory set under the unchanged source clock, Cartesian/closest-feasible rules, joint limits, collision classifier and temporal bounds. This is NOT a mathematical proof that every possible common solver must fail. In particular, finite SLSQP/trust-region failures do not certify global temporal infeasibility. No source corruption, real hardware or destructive action is involved.\n\n'
    text+='The previous natural-start boundary conflict was addressed with the explicitly authorized common preparation convention. All 22 TRAIN11 first-target preparation paths passed sampled kinematic checks, with a provisional 0.733333333s common duration. Final source-start configurations, their preparation joins and full TRAIN11 source execution have not been jointly qualified, so preparation is not finally frozen.\n\n'
    text+='A00 and B00/B24/B49 now have complete source-position trajectories satisfying numeric, detailed-collision, limit and temporal gates. A24/A49 remain unqualified after common propagation, framewise witness continuation, whole-trajectory restoration, fixed certified anchors, sparse-precision retries, local dense retries and independent hard-window searches. The completed dense Cartesian forensic was reused, not rerun.\n\n'
    text+='## Exact remaining source-position evidence\n\n'
    for r in arows:
        if r['source_position_qualified']:continue
        m=r['metrics'];tm=m['temporal']
        text+=f"- {r['case']}: Cartesian failing frames {m['failed_frames']}; qdot {tm['maximum_velocity_rad_s']:.9f} rad/s (limit4.5); qddot {tm['maximum_acceleration_rad_s2']:.9f} rad/s² (limit130); step L2 {tm['maximum_step_norm_rad']:.9f} rad (limit0.179). Candidate: {r['trajectory']['path']}; SHA256 {r['trajectory']['sha256']}.\n"
    text+='\nSource-frame raw fidelity and executable validity remain separate. A has294/2066 certified-unreachable frames. B has30/2066, all in B00; B49 has none. The earlier four-case forensic excluded B00 because its trajectory passed the old95% rule. It did not establish that every B target was reachable. The same analytic hinge-orbit lower bound and the same numerical rule are now applied to all six trajectories.\n\n'
    text+='The shared numerical slack is0.01mm: the existing10μm geometry numerical-resolution floor dominates the measured deterministic TRAIN repeatability. It was not enlarged after A failures. The remaining search uncertainty is solver convergence versus joint-temporal feasibility; only framewise raw infeasibility has analytic certificates.\n\n'
    text+='Continuing to 6D, loaded articulation, datasets, policy training or physical outcomes would bypass an unqualified scientific prerequisite. No physical success result is inferred from source-position passes. No A-specific target, timing, environment or threshold was introduced.\n'
    atomic_text(blocker,text);atomic_text(MASTER/'LATEST_BLOCKER.md',text)
    next_report=MASTER/'NEXT_ACTION_REPORT.md'
    atomic_text(next_report,'# Next scientifically valid action\n\nResume from the persisted common trajectory candidates, not from the old startup blocker. Preserve the four accepted source trajectories and all raw targets. The remaining target is a complete common executable A24/A49 source trajectory under unchanged gates, followed by preparation joins and full TRAIN11 qualification. A further bounded common solver/certification method must distinguish temporal infeasibility from failure to converge; finite local optimizer failure alone is not a certificate.\n\nDo not silently use the2mm slack cap as a success-tuned tolerance. A common source-clock dilation or allowing trajectory-level correction of framewise-reachable targets would change the currently authorized scientific contract and needs an explicit scientific amendment; neither was applied. Do not rerun the602-frame dense forensic. The four previously uncertified frames now have analytic orbit certificates.\n\nOnly after common position passes: qualify6D, loaded Dex3, reference timing/articulation, freeze execution, perform exact action/data audit, regenerate and retrain both only if required, validate policies, run fixed DEV smoke, freeze, then execute matched DEV35 once with measured PhysX traces.\n')
    # Audit method-blind numerical cores without treating presentation labels as execution branches.
    cores=[ROOT/'tools'/x for x in ('master_autonomous_common.py','common_constrained_continuation.py','common_fixed_anchor_continuation.py','common_fixed_anchor_precision.py','common_fixed_anchor_dense.py','common_wrist_nullspace_search.py','common_hinge_orbit_bounds.py','common_grouped_workspace_bounds.py','common_hard_window_search.py','common_trust_window_search.py')]
    audit=[]
    for p in cores:
        tree=ast.parse(p.read_text());names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        forbidden=names & {'representation_mode','episode','case','task_success','physical_outcome','object_pose','method_id'}
        assert not forbidden,(p,forbidden);audit.append(dict(implementation=file_record(p),forbidden_identifiers=[]))
    atomic_json(MASTER/'COMMON_CORE_METHOD_BLIND_AUDIT.json',dict(rows=audit,core_audit_pass=True,full_pipeline_qualified=False,dataset_parity='NOT_AUDITED'))
    parity=OUT/'02_common_execution_qualification/FINAL_COMMON_PIPELINE_PARITY_AUDIT.md'
    if parity.exists() and not (MASTER/'provenance/FINAL_COMMON_PIPELINE_PARITY_AUDIT.md').exists():atomic_text(MASTER/'provenance/FINAL_COMMON_PIPELINE_PARITY_AUDIT.md',parity.read_text())
    atomic_text(parity,'# Common pipeline parity audit — autonomous continuation\n\nThe only intended representation difference remains WRIST versus INTERACTION raw spatial target generation. Raw target bundles, task registration, source-relative events, G1/Dex3 model inputs, limits and10μm detailed geometry convention retain their verified hashes. Numerical cores accept no representation/episode/task-success identity. Candidate recovery is driven by residual, geometry and continuity validity.\n\nThe complete common execution pipeline is NOT QUALIFIED: source-position passes A1/3,B3/3; preparation joins and TRAIN11 source gate incomplete. No claim of zero corrected-dataset confounds or identical new ACT training is made because those stages have not run. See master_autonomous/COMMON_CORE_METHOD_BLIND_AUDIT.json and the bounded diagnostic manifest.\n')
    final=OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md'
    old=MASTER/'provenance/FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md'
    assert old.exists(),'Prior report must remain preserved'
    headings=[
      ('Scientific variable and prior provenance','A=WRIST-trajectory-centric raw targets; B=INTERACTION-centric raw targets. All downstream handling is intended to remain common and method-blind. Earlier physical runs remain PRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY; none is mixed into this continuation.'),
      ('Authoritative Cartesian forensic','The602-frame preceding dense forensic and634 recorded artifacts were verified and reused. Mixed solver-tracking failure and raw target-embodiment infeasibility. The new analytic hinge-orbit certificate resolves all four previously uncertified A frames. It supplements actual FK/IK witnesses and never declares reachability merely from an outer bound.'),
      ('Full SMOKE3 raw feasibility','A:1772/2066 known reachable;294/2066 certified unreachable. B:2036/2066 known reachable;30/2066 certified unreachable, all in B00. B49 remains686/686 reachable. Full-frame accounting exposes the30 B00 failures previously tolerated by the old95% trajectory gate. These are TRAIN diagnostics, not DEV35 outcomes.'),
      ('Task registration and event clock','Protected task/workspace registration and source target arrays were unchanged. The original common source event timestamps and Dex3 semantic arrays are retained. A preparation prefix shifts global execution origin only; no source-event dilation or method-specific timing was applied.'),
      ('Natural-start preparation','Natural G1 ARM14 q0 is exactly preserved. Dex3 uses the explicitly requested common OPEN vector, which differs from the legacy initial hand pre-shape; full28-state equality is NOT claimed. See STARTUP_STATE_SEMANTICS_AUDIT.json.22/22 TRAIN11 pointwise first-target preparation paths passed sampled limits, geometry, velocity and acceleration tests. Provisional common duration:0.733333333s,22 intervals/23 endpoint samples at30Hz. This is the maximum analytical minimum within the declared rest-to-rest quintic joint-path family, not a global time-optimal planning claim. Final source joins and duration freeze remain pending. No standardized-grasp initialization was used.'),
      ('Common morphology realization','Raw positions are saved separately from executable q/FK, position corrections, certified lower bounds and optimality gaps. Common numerical slack0.01mm comes from the existing10μm geometry-resolution floor and deterministic TRAIN repeatability; the user2mm cap was not substituted as a permissive gate. A294-frame raw correction lower mean/p95/max: '+str(gap['raw_position_correction_lower_mm'])+' mm; valid witness upper: '+str(gap['valid_witness_upper_mm'])+' mm. These bounds concern exact raw-position correction, not correction only to enter the10mm acceptance region. B30-frame bounds are recorded separately in B_CERTIFIED_MORPHOLOGY_GAP.json.'),
      ('Common position recovery and qualification','Source-position passes: A1/3 (A00), B3/3. Passing source trajectories were checked for detailed hard collisions0, unresolved geometry0, hard limits0, finite states, branch/step/velocity/acceleration validity. A24/A49 remain unqualified; exact frames and residuals are in FRAMEWISE_DIAGNOSTICS.csv and the blocker report. Full prepared trajectories and the full TRAIN11 source gate did not pass. Completed bounded families and45 independent hard-window trials are retained; some relaxed windows have witnesses, but no global temporal infeasibility proof is claimed.'),
      ('Full6D qualification','NOT RUN: the common complete position prerequisite is not satisfied. No orientation calibration change or method-specific rescue was introduced.'),
      ('Loaded Dex3 qualification','NOT RUN. No new14/14 mapping/sign/readback/runtime-limit pass or measured-limit compliance is claimed. Existing isolated tests are not promoted to loaded qualification.'),
      ('Shared reference pipeline and execution freeze','NOT QUALIFIED. The requested03_common_execution_freeze manifest is not produced as a false pass. Diagnostic snapshots are hashed but are not a qualified final execution freeze.'),
      ('Exact old-versus-corrected action/dataset audit','NOT RUN: no complete qualified executable reference dataset exists. Exact changed episodes/frames/scalars, timing, normalization and chunk differences are not available. No retraining decision is inferred from metadata alone.'),
      ('Dataset regeneration and parity','No corrected datasets generated. UNINTENDED_DATASET_CONFOUNDS is NOT AUDITED, not asserted0.'),
      ('ACT training and checkpoint selection','No new ACT training/retraining or checkpoint selection. Existing checkpoints remain unchanged provenance; none is promoted to a frozen final A/B evaluation checkpoint. A common future selection rule must be declared before training/DEV outcomes.'),
      ('Policy sanity and DEV35 structural smoke','NOT RUN. DEV35 remains repeatedly inspected development data. No policy-interface or complete physical-trace pass is fabricated.'),
      ('DEV35 physical stage results and TSR','NOT RUN: new physical rollouts executed A0,B0; success counts, percentages and stage outcomes are NOT AVAILABLE, not0% failure rates.'),
      ('Matched statistics','NOT AVAILABLE. No matched outcome matrix, confidence interval, bootstrap estimate or McNemar result without physical runs.'),
      ('Figures, tables and replays','Valid TRAIN diagnostic artifacts: '+str(diag)+'. Includes a per-frame CSV, diagnostic table, source-residual/continuity figure, raw-feasibility/source-gate figure and certified morphology-gap figure in PNG/PDF/SVG. No DEV35 main figure/table or actual PhysX35-split videos were generated because no physical traces exist in this rebuild. No command-only animation is presented as physics evidence.'),
      ('Fresh-test readiness','NOT READY for frozen untouched final-test evaluation. DEV35 must not be relabeled unseen. READY_FOR_UNTOUCHED_FINAL_TEST.md documents the required qualification/freeze/collection procedure; no fresh manifest is fabricated.'),
      ('Limitations and exact paper-safe interpretation','The evidence supports a TRAIN-side comparison of raw morphology feasibility and documents partial common solver recovery. It does not establish learned-policy superiority, DEV35 task success, untouched-test performance, target-domain visual deployment or real-G1 success. The remaining solver convergence versus temporal feasibility question is unresolved; local search failure is not a theorem about the representation.'),
      ('Stop and resume','True stop condition4 is used narrowly for failure to construct a complete accepted common trajectory set after the documented bounded realization. It is not a universal mathematical impossibility claim. Preserve the four qualified source trajectories and all diagnostics. See NEXT_ACTION_REPORT.md. No targets, clock, collision tolerance or limits were relaxed to force completion.')]
    body='# Final autonomous single-variable A/B rebuild report\n\n'+TERMINAL+'\n\nDiagnostic report ready; final DEV35 results are NOT ready.\n\nDiagnostic snapshot SHA256: '+d['diagnostic_snapshot_sha256']+'\n\n'
    for i,(title,content) in enumerate(headings,1):body+=f'## {i}. {title}\n\n{content}\n\n'
    atomic_text(final,body)
    ready=OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md';archive=MASTER/'provenance/READY_FOR_UNTOUCHED_FINAL_TEST.md'
    if ready.exists() and not archive.exists():atomic_text(archive,ready.read_text())
    atomic_text(ready,'# Untouched final-test readiness\n\nREADY_FOR_UNTOUCHED_FINAL_TEST_EVALUATION = NO\n\nA complete common executable trajectory pipeline has not passed. No final execution/dataset/checkpoint/scorer freeze is certified in this continuation. DEV35 remains development data.\n\nAfter completing common position,6D,loaded Dex3,reference smoke,exact action audit,paired dataset/training parity and DEV smoke: freeze all executable files, data, checkpoints, environments, scorer, cameras and membership. Collect a genuinely new FINAL_TEST manifest after that freeze, never use those episodes for calibration/debugging/checkpoint selection, and evaluate the same matched frozen pipeline once. Keep normal failures; rerun only infrastructure-invalid cases. Save measured robot and actual PhysX object states. No new test set or unseen performance is fabricated here.\n')
    _,mapping=verified_oracle_contract();atomic_json(MASTER/'REPORT_PUBLICATION_PROVENANCE.json',dict(report_only_substitutions=mapping,execution_hash_substitutions=0,previous_report=file_record(old),current_report=file_record(final)))
    korean='과학적 중단 조건4: 여러 공통 유한 복구를 완료했으나 전체 실행 가능 궤적을 구성하지 못했습니다. 소스 위치 통과는 A1/3,B3/3이며 A24/A49의 위치·시간 제약이 남았습니다. 이는 전역적 불가능성의 수학적 증명이 아닙니다. 자연 q0와 타깃·등록·이벤트·충돌·관절 한계를 유지했고 A/B별 예외는 없습니다. A294/2066,B30/2066 원시 프레임의 도달 불가능성을 인증했습니다. 진단 표·그림·프레임별 데이터와 최종 차단 보고서를 저장했습니다.6D·Dex3·학습·DEV35 물리 평가는 실행하지 않았으며 CLI는 이 과학적 경계에서 중단합니다.'
    log('BOUNDED_COMMON_EXECUTION_FINAL_HANDOFF',TERMINAL,[diag/'DIAGNOSTIC_MANIFEST.json',blocker],
        'A24/A49 remain unqualified under unchanged position and temporal gates after completed bounded common recovery',
        'BOUNDED_COMMON_TRAJECTORY_REALIZATION_NOT_QUALIFIED',
        'Preserved qualified source trajectories, exact failed frames, analytic morphology certificates, diagnostic figures/tables and a paper-safe blocker report. No downstream prerequisite bypass.',
        [final,blocker,next_report,ready,diag/'DIAGNOSTIC_MANIFEST.json'],'FURTHER_COMMON_TRAJECTORY_FEASIBILITY_WORK_REQUIRED',korean,6)
    atomic_text(OUT/'CURRENT_STATUS.md','# Current authoritative state\n\n'+TERMINAL+'\n\nSource-position: A1/3,B3/3. A24/A49 remain unqualified after bounded common recovery. Natural-start preparation paths22/22 pass provisionally; final joins/TRAIN11 source gate are incomplete. Raw certified-unreachable frames: A294/2066,B30/2066. No6D,Dex3,training orDEV35 physics. See master_autonomous/CURRENT_STAGE.md and FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md.\n')
    atomic_json(OUT/'CURRENT_STATUS.json',dict(status=TERMINAL,source_position_counts=d['source_position_counts'],prepared_pipeline_qualified=False,full_6d='NOT_RUN',loaded_dex3='NOT_RUN',DEV35='NOT_RUN',report=file_record(final)))
    required=[OUT/'03_common_execution_freeze/COMMON_EXECUTION_FREEZE_MANIFEST.json',*[OUT/'07_paper_artifacts'/p for p in ('TABLE_SINGLE_VARIABLE_AB_DEV35_RESULTS.md','TABLE_SINGLE_VARIABLE_AB_DEV35_RESULTS.csv','FigXX_Single_Variable_AB_DEV35_double.png','FigXX_Single_Variable_AB_DEV35_double.pdf','FigXX_Single_Variable_AB_DEV35_double.svg','FIGURE_RAW_FEASIBILITY_AB.png','FIGURE_STAGE_SUCCESS_AB.png','FIGURE_MATCHED_EPISODE_MATRIX_AB.png')],*[OUT/p for p in ('A_DEV35_FINAL_TOP_35SPLIT.mp4','A_DEV35_FINAL_OVERVIEW_35SPLIT.mp4','B_DEV35_FINAL_TOP_35SPLIT.mp4','B_DEV35_FINAL_OVERVIEW_35SPLIT.mp4')]]
    atomic_json(MASTER/'REQUIRED_FINAL_ARTIFACT_AUDIT.json',dict(status='FINAL_RESULTS_NOT_READY',records=[dict(path=str(p),exists=p.exists(),valid_for_this_rebuild=False,reason='Prerequisite common execution not qualified') for p in required]))
    atomic_json(MASTER/'FINAL_RESULT.json',dict(status=TERMINAL,stop_condition=4,global_temporal_infeasibility_proven=False,
        source_position_counts=d['source_position_counts'],raw_certified_counts={'WRIST':294,'INTERACTION':30},raw_frame_denominators={'WRIST':2066,'INTERACTION':2066},
        prepared_pipeline_qualified=False,final_report=file_record(final),diagnostic_manifest=file_record(diag/'DIAGNOSTIC_MANIFEST.json'),DEV35_physical_results=None))
    atomic_json(MASTER/'COMMON_POSITION_BOUNDED_RUN_MANIFEST.json',dict(qualified_full_execution=False,
        artifacts=[file_record(p) for p in sorted(RUN.rglob('*')) if p.is_file()],
        drivers=[file_record(p) for p in sorted((ROOT/'tools').glob('run_autonomous_*.py'))]))
    artifacts=[p for p in MASTER.rglob('*') if p.is_file() and p.name!='MASTER_ARTIFACT_MANIFEST.json']
    artifacts.extend([final,ready,parity,OUT/'CURRENT_STATUS.md',OUT/'CURRENT_STATUS.json'])
    atomic_json(MASTER/'MASTER_ARTIFACT_MANIFEST.json',dict(status=TERMINAL,qualified_execution_freeze=False,artifacts=[file_record(p) for p in sorted(set(artifacts))],core_implementations=[file_record(p) for p in cores]))
    # Verify every published record, including generated figures, before handoff.
    manifest=read(MASTER/'MASTER_ARTIFACT_MANIFEST.json')
    assert all(file_record(Path(r['path']))==r for r in manifest['artifacts']+manifest['core_implementations'])
    verified_oracle_contract()
    print('\n==================================================\nMASTER ALOHA→G1 SINGLE-VARIABLE A/B RESULTS\n==================================================\n')
    print('STARTUP PREPARATION\nNatural G1 ARM14 q0 preserved: YES\nDex3: explicit common OPEN; legacy hand pre-shape is not reused\nCommon preparation duration: 0.733333333s PROVISIONAL; final source joins not qualified\nPreparation identical A/B: SAME RULE AND PROVISIONAL DURATION\n')
    print('RAW CARTESIAN FEASIBILITY\nA reachable:1772/2066\nA certified unreachable:294/2066\nA correction lower mean/p95/max mm:',gap['raw_position_correction_lower_mm'])
    print('B reachable:2036/2066\nB certified unreachable:30/2066\nB correction bounds:',bgap,'\n')
    print('EXECUTABLE POSITION\nA qualified SOURCE trajectories:1/3\nB qualified SOURCE trajectories:3/3\nFull preparation+source gate:NOT QUALIFIED\nHard collision:0 in qualified source trajectories; other candidates not promoted\nHard limits:0\nBranch/temporal discontinuity:remaining A24/A49 failures documented\n')
    print('FULL6D:NOT RUN\nDEX3 mapping/sign/readback/runtime limits/measured violations:NOT RUN\nDATASET action diff/regeneration/parity:NOT RUN\nACT retraining decision:NOT AUDITED\nACT-A/B checkpoints:no new selection or training\n')
    print('DEV35\nA physical rollouts executed:0/35\nB physical rollouts executed:0/35\nApproach/Grasp/Lift/Handoff/Ownership/Transport/Bin/FULL TASK:NOT AVAILABLE\nB-A:NOT AVAILABLE\nMcNemar:NOT AVAILABLE\n')
    print('MAIN DEV35 FIGURE:NOT GENERATED\nDEV35 RESULT TABLE:NOT GENERATED\nA TOP/OVERVIEW VIDEOS:NOT GENERATED\nB TOP/OVERVIEW VIDEOS:NOT GENERATED\nTRAIN DIAGNOSTIC FIGURE:',diag/'FIGURE_TRAIN_POSITION_DIAGNOSTIC.png')
    print('TRAIN DIAGNOSTIC TABLE:',diag/'TABLE_TRAIN_POSITION_DIAGNOSTIC.md','\nFINAL REPORT:',final,'\nCHATGPT UPDATE:',MASTER/'CHATGPT_UPDATE.md')
    print('==================================================\n'+TERMINAL,flush=True)


if __name__=='__main__':run()
