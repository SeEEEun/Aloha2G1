#!/usr/bin/env python3
"""Verify supported final artifacts and write a paper-safe, hash-bound report."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_position_run import OUT,DEST,FREEZE,verify,read,file_record,atomic_json,atomic_text,now
PAPER=OUT/'07_paper_artifacts/final'
REPORT=OUT/'FINAL_PAPER_READY_SINGLE_VARIABLE_AB_REPORT.md'

def main():
    contract=verify();r=read(PAPER/'FINAL_NUMERICAL_RESULTS.json');replays=read(PAPER/'REPLAY_MANIFEST.json');act=read(DEST/'ACT_BRANCH_RESULT.json');audit=read(DEST/'action_dataset_audit/EXACT_ACTION_DATASET_AUDIT.json')
    from tools.final_paper_fairness_audit import main as verify_parity
    verify_parity()
    if act['status'] not in ('UNAVAILABLE_EMPTY_PAIRED_EXECUTABLE_TRAIN_SET','COMPLETE','UNAVAILABLE_IRRECOVERABLE_TECHNICAL_ISSUE'):raise RuntimeError('ACT branch must be completed or explicitly unavailable, not silently abandoned')
    required=['TABLE_1_RETARGETING_FEASIBILITY.md','TABLE_1_RETARGETING_FEASIBILITY.csv','TABLE_2_REFERENCE_DEV35_PHYSICAL.md','TABLE_2_REFERENCE_DEV35_PHYSICAL.csv','Fig_Main_SingleVariable_AB.png','Fig_Main_SingleVariable_AB.pdf','Fig_Main_SingleVariable_AB.svg','Fig_Raw_Cartesian_Feasibility_AB.png','Fig_Morphology_Correction_Distribution.png','Fig_DEV35_Stage_Success_AB.png','Fig_Matched_DEV35_Matrix.png']
    required += ['Fig_DEV35_First_Failure_Stage.png','Fig_Representative_Physical_Rollouts.png','Fig_TRAIN_Frozen_Solver_Bottleneck.png','TRAIN_BOTTLENECK_DIAGNOSTIC_MANIFEST.json','MATCHED_DEV35_EPISODE_OUTCOMES.csv','MATCHED_DEV35_EPISODE_OUTCOMES.md']
    required += ['DETAILED_GEOMETRY_FAILURE_REASON_AUDIT.json','MEDIA_DECODE_VERIFICATION.json','FINAL_COHORT_INTEGRITY_AUDIT.json']
    if (PAPER/'VISUAL_REVIEW.json').exists():required.append('VISUAL_REVIEW.json')
    paths=[PAPER/name for name in required]
    paths.extend(Path(x['artifact']['path']) for x in replays['products'].values())
    assert len(replays['products'])==4
    assert all(p.is_file() and p.stat().st_size>0 for p in paths)
    for name in ('MEDIA_DECODE_VERIFICATION.json','FINAL_COHORT_INTEGRITY_AUDIT.json','VISUAL_REVIEW.json'):
        assert read(PAPER/name)['status']=='PASS'
    for name,row in replays['products'].items():
        assert file_record(Path(row['artifact']['path']))==row['artifact']
        assert row['probe']['width']==3840 and row['probe']['height']==2160 and row['probe']['avg_frame_rate']=='30/1' and row['probe']['codec_name']=='h264'
        assert replays['common_video_frames']<=int(row['probe']['nb_frames'])<=replays['common_video_frames']+1
    stats=r['reference_dev35'];feas=r['feasibility'];counts=stats['counts'];a=counts['WRIST']['FULL_TASK_SUCCESS'];b=counts['INTERACTION']['FULL_TASK_SUCCESS']
    available=stats['actual_physics_trace_counts'];dex=read(DEST/'dex3/COMMON_DEX3_QUALIFICATION.json')
    clockpath=DEST/'dex3/SOURCE_CLOCK_QUALIFICATION.json';clock=read(clockpath) if clockpath.exists() else {'status':'NOT_AVAILABLE'}
    assert dex['status']=='PASS' and clock['status']=='PASS'
    paired=read(DEST/'action_dataset_audit/PAIRED_TRAIN_SET_MANIFEST.json')
    files=[file_record(p) for p in paths]+[file_record(FREEZE),file_record(OUT/'03_common_execution_freeze/COMMON_FULL6D_FREEZE_MANIFEST.json'),file_record(PAPER/'FINAL_NUMERICAL_RESULTS.json'),file_record(PAPER/'REPLAY_MANIFEST.json'),file_record(DEST/'ACT_BRANCH_RESULT.json'),file_record(DEST/'action_dataset_audit/EXACT_ACTION_DATASET_AUDIT.json'),file_record(DEST/'FINAL_COMMON_PIPELINE_PARITY_AUDIT.json')]
    freeze_files=[file_record(p) for p in sorted((OUT/'03_common_execution_freeze').glob('*.json')) if p.name.startswith('COMMON_')]
    fresh=OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md'
    atomic_text(fresh,'# Fresh untouched final-test protocol\n\nDEV35 is repeatedly inspected development data, not an unseen final test. No fresh results are fabricated.\n\n1. Collect new source demonstrations after the complete pipeline freeze. Predeclare membership and recording IDs before inspecting retargeting or physical outcomes. Exclude every prior source recording used for solver work, image diagnosis, training, checkpoint choice or controller/scorer tuning.\n2. Preserve and hash original RGB, state, action and timestamps. Apply the same frozen source event extraction, task registration and representation builders. Generate one source-derived environment per episode, shared exactly by A/B.\n3. Feed the new immutable manifest to the same frozen numerical functions and their fixed budgets. Do not change q0,0.700s preparation, joint limits, physical temporal constraints, geometry tolerance, target conventions, controllers or scorer. No individual rescue.\n4. Run each matched frozen reference once. Non-executable trajectories remain end-to-end failures with the full fixed denominator. Persist actual measured robot/object/contact traces for executed runs. Repair/rerun only independently documented infrastructure-invalid attempts.\n5. If a completed paired ACT experiment exists, use its already selected fixed checkpoints and source-conditioning interface unchanged. Otherwise report ACT unavailable; do not silently reuse incompatible old checkpoints.\n6. Publish exact paired counts, confidence intervals, McNemar results and all first failures. Do not tune after opening the fresh outcomes.\n\nFrozen manifests:\n'+''.join(f'- `{x["path"]}` SHA256 `{x["sha256"]}`\n' for x in freeze_files)+'\nReadiness means the fixed experimental protocol can be applied to fresh data; it is not a claim of task success, unseen performance, autonomous target-domain perception, or real-G1 execution.\n')
    text='# Final single-variable ALOHA → G1 paper experiment\n\n'
    text+=f'Generated: {now()}\n\n**DEV35 DEVELOPMENT EVALUATION — not an untouched final test.**\n\n'
    text+='## Results at a glance\n\nThis is the new uniform frozen procedure, not the prior per-case recovery portfolio.\n\n| Endpoint | A — Wrist | B — Interaction |\n|---|---:|---:|\n'
    for label,group,key,n in [('TRAIN40 executable position','TRAIN40','executable_position',40),('DEV35 executable position','DEV35','executable_position',35),('DEV35 executable 6D','DEV35','full6d_executable',35),('DEV35 complete executable action','DEV35','complete_action_executable',35)]:
        text+=f'| {label} | {feas[group]["WRIST"][key]}/{n} | {feas[group]["INTERACTION"][key]}/{n} |\n'
    text+=f'| Actual DEV35 PhysX trials | {available["WRIST"]} | {available["INTERACTION"]} |\n| Cumulative reference full task | {a}/35 | {b}/35 |\n\n'
    text+=f'ACT: **{act["status"]}**; paired usable training episodes: {len(paired["included_ids"])}.\n\n'
    if not sum(available.values()):text+='No DEV35 episode reached physical execution. The zero cumulative task rates are upstream feasibility outcomes, not measured grasp failures. Videos contain explicit non-execution failure cards.\n\n'
    text+='## 1. Experimental question and single-variable contract\n\nWe compare wrist-trajectory-centric and interaction-centric raw target generation under a shared source-event clock, task registration, bounded G1 realization, Dex3 controller, and contact-constrained simulation. The question is how the representations differ in raw morphology fit, executable trajectory yield, and cumulative manipulation outcomes. The frozen numerical algorithms receive no method identity, episode outcome or task-success objective. A/B labels select raw input arrays and organize reports only.\n\n'
    text+='## 2. Frozen common execution and compute budget\n\n'+json.dumps({k:contract[k] for k in ('selection','budget','acceptance','certification')},indent=2)+'\n\nThe0.179rad aggregate bound is internal only, not final trajectory acceptance. The exact common natural q0 and0.700s/21-frame preparation are retained. All source-relative event times and raw targets remain preserved. Per-joint limits,4.5rad/s,130rad/s², adaptive branch checks and10µm detailed-geometry tolerance are unchanged. Proxy-only overlap is diagnostic; unresolved geometry fails closed.\n\n'
    text+='## 3. Prior TRAIN11 diagnostic work\n\nPrior best-of-bounded-recovery diagnostics achieved SMOKE3 A3/3,B3/3 and TRAIN11 A7/11,B10/11. Those were not produced by one identical per-episode search portfolio. They remain provenance and are not substituted into this new cohort. The current experiment uniformly applies the latest completed TRAIN-side posture-pool procedure with its explicit finite budget, so its yield can differ from the earlier diagnostic portfolio. No extra budget or individual repair follows a scientific failure.\n\n'
    text+='The same predeclared11-episode subset within the new uniform frozen run yields:\n\n'+json.dumps(r['uniform_frozen_train11'],indent=2)+'\n\nThese are diagnostic outcomes, not a mandatory all-pass gate.\n\n'
    text+='## 4. Full TRAIN40 and DEV35 retargeting feasibility\n\n'+(PAPER/'TABLE_1_RETARGETING_FEASIBILITY.md').read_text()+'\n'
    for group in ('TRAIN40','DEV35'):
        text+=f'### {group} exact position classes\n\n'
        for mode in ('WRIST','INTERACTION'):text+=f'- {mode}: `{json.dumps(feas[group][mode]["position_outcomes"],sort_keys=True)}`\n'
        text+='\n'
    text+='Raw lower-bound certificates concern inability to track the unmodified raw Cartesian target within10mm. They do not prove there is no allowed closest-feasible trajectory. A bounded no-witness or colliding candidate does not prove global impossibility. The exact candidate reports, every attempted family and geometry diagnostics are retained. Numerical residual ≤10mm, a geometry-valid framewise witness, and a temporally executable trajectory are reported separately. Executable-correction statistics never silently include rejected configurations.\n\n'
    text+='The model-only outer-envelope certificate is a lower bound, not generally the exact constrained closest-feasible optimum. Joint limits, detailed collision and temporal constraints can make the actual minimum correction larger. Consequently, failure of the frozen near-bound realization criterion is reported only as failure of this fixed procedure, not as proof that no useful physical realization exists.\n\n'
    text+='Matched position and 6D feasibility statistics (exploratory, not untouched-test inference):\n\n```json\n'+json.dumps(r['paired_feasibility_statistics'],indent=2)+'\n```\n\n'
    text+='## 5. Full6D stage\n\nThe orientation-chain audit precedes the run. Registered model wrist matrices are consumed directly without a second TCP calibration. The common existing temporal DLS is seeded from qualified position trajectories and uses its predeclared250/40/25-iteration procedure. Whole-trajectory position, orientation, limit, temporal and geometry checks determine the outcome. Orientation error is not silently projected away or labeled mathematically infeasible. Position failures are not run through6D and remain cumulative failures.\n\n'
    for mode in ('WRIST','INTERACTION'):text+=f'- {mode}: TRAIN40 full6D {feas["TRAIN40"][mode]["full6d_executable"]}/40; DEV35 full6D {feas["DEV35"][mode]["full6d_executable"]}/35. Orientation-attempt residual: `{json.dumps(feas["DEV35"][mode]["orientation_attempt_residual_rad"])}`.\n'
    text+='\nA6D failure is not automatically an orientation failure. The final raw/closest-feasible Cartesian gate, temporal constraints and detailed geometry are rechecked after the common orientation solve. Exact failed-component counts (multiple causes may coexist):\n\n'
    for group in ('TRAIN40','DEV35'):
        for mode in ('WRIST','INTERACTION'):text+=f'- {group} {mode}: `{json.dumps(feas[group][mode]["full6d_failure_component_counts"])}`; final attempted Cartesian residual (mm): `{json.dumps(feas[group][mode]["full6d_attempt_cartesian_residual_mm"])}`.\n'
    text+='\n## 6. Common Dex3 qualification and timing\n\n'+f'Zero-contact/gravity-loaded and qualified contact component status: {dex["status"]}. Source-clock adapter qualification: {clock["status"]}.\n\n'
    text+='The14-joint command/DOF/readback mapping, axis/sign, drive and runtime limits were audited with the existing0.005rad safety inset. Measured states were not clipped. Legacy 1.5s preshape plus1.5s closure after a source trigger was identified by interface inspection as incompatible with the qualified event clock. A common source-clock adapter removes that extra delay while retaining the contact debounce, mechanical confirmation and bounded preload. Receiver confirmation gates actual giver release, which cannot begin before the source release phase. Any delayed acquisition is a physical outcome, not an arm rescue. Component tests and any synthetic stress cases are not counted as DEV35 trials.\n\n'
    text+='The final complete-action gate also checks the actual common physical P14 hand commands with the qualified arm trajectory under the unchanged detailed collision rule. This prevents an arm trajectory qualified with a different reference hand configuration from being silently declared a complete executable action. No arm or target repair is applied at this gate. A post-simulation NumPy-integer JSON serialization error was repaired with native integer conversion; the complete measured stress trace was preserved, and old/new command/event equality was verified separately.\n\n'
    text+='## 7. Reference-level DEV35 results\n\n'+(PAPER/'TABLE_2_REFERENCE_DEV35_PHYSICAL.md').read_text()+'\n'
    text+=f'Actual saved PhysX traces: A {available["WRIST"]}, B {available["INTERACTION"]}. Non-executable cases have explicit failure cards, not fabricated simulations. All cumulative end-to-end rates use35. A full task: {a}/35={100*a/35:.1f}%; B: {b}/35={100*b/35:.1f}%; B−A={stats["difference_percentage_points"]:+.1f} percentage points.\n\n'
    if not sum(available.values()):text+='**No reference episode reached physical execution. These are upstream feasibility failures in an end-to-end metric, not observed grasp failures or evidence about physical manipulation quality among executable trajectories.**\n\n'
    text+='## 8. Matched statistics\n\n'+json.dumps(stats,indent=2)+'\n\nClopper–Pearson intervals describe each success proportion. The paired bootstrap uses10,000 shared-episode resamples with seed1000. An all-zero paired sample may give a degenerate empirical bootstrap interval; that does not establish population equivalence. No significant advantage is claimed without supporting discordant outcomes.\n\n'
    text+='## 9. Exact action/dataset audit and paired training subset\n\n'+f'Retraining required: {audit["retraining_required"]}. Old checkpoints reusable: {audit["old_checkpoints_reusable"]}. Paired executable training IDs: `{paired["included_ids"]}` ({len(paired["included_ids"])}/40). Every excluded episode and reason is in the paired manifest.\n\n'
    text+='The old datasets use a method-derived previous-action28-D state surrogate, not identical source-state observations. A corrected ACT branch requires identical source conditioning for both policies and explicit new supervision including the preparation prefix; incompatible old checkpoints/statistics are not reused. Removed episodes/frames/scalars, overlapping numerical diffs where valid targets exist, and exact tensor hashes are in `paper_completion_v1/action_dataset_audit/EXACT_ACTION_DATASET_AUDIT.json`. Missing valid targets are reported as exclusions, never imputed from invalid candidates.\n\n'
    text+='| Method | Excluded old episodes | Excluded old frames | Excluded action scalars | Comparable valid numerical difference |\n|---|---:|---:|---:|---|\n'
    for mode in ('WRIST','INTERACTION'):
        old=audit['methods'][mode];episodes=old['episodes'];removed_frames=sum(x.get('removed_frames',0) for x in episodes);removed_scalars=sum(x.get('removed_action_scalars',0) for x in episodes)
        text+=f'| {mode} | {old["removed_episodes"]} | {removed_frames} | {removed_scalars} | '+('N/A — no paired valid target; not zero difference' if not paired['included_ids'] else 'See exact per-episode audit')+' |\n'
    text+='\nExclusion counts are membership changes, not a claimed elementwise numerical difference against nonexistent valid tensors. No corrected normalization or action chunks were fabricated when the paired set was empty.\n\n'
    text+='## 10. Downstream ACT experiment\n\n'+json.dumps(act,indent=2)+'\n\n'
    if act['status']!='COMPLETE':text+='ACT unavailable does not become a synthetic0/35 policy result. The primary frozen retargeting/reference experiment is reported independently. Prior ACT physical runs and standardized-grasp experiments remain PRE_FINAL_SINGLE_VARIABLE_REBUILD_DIAGNOSTIC_ONLY and are not mixed into these results.\n\n'
    text+='## 11. Figures, tables and replay provenance\n\n'+''.join(f'- `{x["path"]}` — SHA256 `{x["sha256"]}`\n' for x in files)+'\n'
    text+='All videos are7×5,3840×2160,30FPS,H.264 with identical episode ordering/cameras. Actual executions use measured robot state and PhysX object state. Non-executable cases are labeled static failures. Objective representative selection uses the lowest-index case in each observed paired category and the largest certified morphology-gap contrast, not visual preference.\n\n'
    text+='## 12. Limitations and paper-safe interpretation\n\nThe study is conditional on this frozen bounded solver and its TRAIN-derived seed bank. Failure to find a qualified trajectory is not a mathematical impossibility claim. TRAIN11 and DEV35 were repeatedly inspected during prior development; this run does not erase that exposure. Raw fidelity, target-embodiment feasibility, policy learning and physical outcome are distinct endpoints. Preparation and the source-conditioned scene/clock are supplied, not inferred autonomously from G1-domain vision. No real-G1 command was executed. Negative task rates or a weak paired test do not support an assertion of representation superiority.\n\n'
    text+=f'Under the declared fixed common procedure, executable-position DEV35 yield is A {feas["DEV35"]["WRIST"]["executable_position"]}/35 and B {feas["DEV35"]["INTERACTION"]["executable_position"]}/35; cumulative reference full-task yield is A {a}/35 and B {b}/35. These statements describe this development experiment only. Any wrist-reconstruction or whole-hand-interaction advantage beyond the reported measurements remains unproven.\n\n'
    text+='## 13. Frozen SHA256 provenance and untouched-test protocol\n\n'+''.join(f'- `{x["path"]}` SHA256 `{x["sha256"]}`\n' for x in freeze_files)+'\nSee `READY_FOR_UNTOUCHED_FINAL_TEST.md`. No untouched final-test results are claimed.\n'
    # Reporting-only formatting; no numerical or acceptance changes.
    for payload in (json.dumps({k:contract[k] for k in ('selection','budget','acceptance','certification')},indent=2),json.dumps(r['uniform_frozen_train11'],indent=2),json.dumps(stats,indent=2),json.dumps(act,indent=2)):
        text=text.replace(payload,'```json\n'+payload+'\n```')
    text += '\nThe supplementary TRAIN bottleneck figure uses the lowest-index predeclared TRAIN11 case meeting its recorded diagnostic category. It shows saved kinematic candidates and final acceptance failures, not physical execution, and did not trigger any solver modification.\n'
    text+='\nMeasured wall times are descriptive concurrent-run timings, not a controlled hardware speed benchmark. The search/evaluation budget, rather than a wall-clock timeout, was held common.\n'
    atomic_text(REPORT,text)
    verification=dict(status='FINAL_PAPER_RESULTS_READY',generated_at=now(),report=file_record(REPORT),fresh_test_protocol=file_record(fresh),artifacts=files,
        required_artifacts_exist=True,replay_dimensions_verified=True,retargeting_freeze_verified=True,act_branch=act['status'],reference_statistics=stats)
    atomic_json(DEST/'FINAL_ARTIFACT_VERIFICATION.json',verification)
    atomic_text(DEST/'CURRENT_STAGE.md','# Final paper experiment\n\nFINAL_PAPER_RESULTS_READY\n\n'+str(REPORT)+'\n')
    korean=f'최종 논문 산출물 검증이 완료되었습니다. DEV35는 개발 평가입니다. 실제 물리 실행 수는 A {available["WRIST"]}, B {available["INTERACTION"]}이며 전체 과제 누적 성공은 A {a}/35, B {b}/35입니다. 비실행 사례는 명시적 실패 카드로 표시했고, ACT 상태는 {act["status"]}입니다. 원시 타깃·실행 가능성·물리 결과를 분리해 보고했습니다.\n'
    atomic_text(DEST/'CHATGPT_UPDATE.md',korean);atomic_text(OUT/'CURRENT_STATUS.md','# Final paper experiment\n\nFINAL_PAPER_RESULTS_READY\n\n'+str(REPORT)+'\n\n'+korean)
    master=OUT/'master_autonomous'
    atomic_text(master/'CURRENT_STAGE.md','# Final paper experiment\n\nFINAL_PAPER_RESULTS_READY\n\n'+str(REPORT)+'\n')
    atomic_text(master/'CHATGPT_UPDATE.md',korean)
    atomic_json(master/'CHECKPOINT_STATE.json',dict(status='FINAL_PAPER_RESULTS_READY',run=str(DEST),verification=file_record(DEST/'FINAL_ARTIFACT_VERIFICATION.json')))
    log=dict(timestamp=now(),stage='FINAL_PAPER_ARTIFACT_VERIFICATION',status='FINAL_PAPER_RESULTS_READY',authoritative_inputs=files,files_hashes_used=files,observed_problem=None,root_cause_classification='COMPLETED_FROZEN_EXPERIMENT',action_taken='Verify all supported paper results and explicit optional-branch status',retries=0,artifacts_created=[file_record(REPORT),file_record(DEST/'FINAL_ARTIFACT_VERIFICATION.json')],next_stage='FRESH_UNTOUCHED_FINAL_TEST_COLLECTION')
    for folder in (DEST,master):
        with (folder/'MASTER_RUN_LOG.jsonl').open('a') as stream:stream.write(json.dumps(log)+'\n')
    print('============================================================\nFINAL SINGLE-VARIABLE ALOHA→G1 PAPER RESULTS\n============================================================\nFROZEN COMMON PIPELINE: PASS\nTRAIN11 DIAGNOSTIC: A7/11, B10/11 (prior diagnostic portfolio)')
    print('Uniform frozen TRAIN11:',{m:f"{v['executable']}/11" for m,v in r['uniform_frozen_train11'].items()})
    for group in ('TRAIN40','DEV35'):
        print(group,'RETARGETING')
        for mode in ('WRIST','INTERACTION'):
            f=feas[group][mode];print(mode,f['executable_position'],'/',f['episodes'],'position; certified raw-unreachable frames',f['certified_raw_unreachable_frames'],'/',f['frames'],'; certified complete-trajectory infeasibility',f['certified_executable_trajectory_infeasibility_count'])
        print('B-A executable position:',100*(feas[group]['INTERACTION']['executable_position']-feas[group]['WRIST']['executable_position'])/feas[group]['WRIST']['episodes'],'percentage points')
    print('------------------------------------------------------------\nREFERENCE PHYSICAL DEV35\n                      A          B')
    order=[('Executable','EXECUTABLE_TRAJECTORY'),('Approach','APPROACH_VALID'),('Grasp','LEFT_GRASP_SUCCESS'),('Lift','LIFT_SUCCESS'),('Handoff','HANDOFF_SUCCESS'),('Ownership','RIGHT_OWNERSHIP_SUCCESS'),('Transport','RIGHT_TRANSPORT_SUCCESS'),('Bin','BIN_ENTRY_SUCCESS'),('Settle','BIN_SETTLE_SUCCESS'),('Full Task','FULL_TASK_SUCCESS')]
    for label,s in order:print(f'{label:18} {counts["WRIST"][s]:2}/35      {counts["INTERACTION"][s]:2}/35')
    print('Actual PhysX trials:',available,'; upstream failures are not observed physical failures.')
    print('B-A full-task difference:',stats['difference_percentage_points'],'percentage points\nExact full-task McNemar:',stats['exact_mcnemar_pvalue'])
    print('------------------------------------------------------------\nACT\nRetraining required:', 'YES' if audit['retraining_required'] else 'NO','\nPaired TRAIN episodes:',len(paired['included_ids']))
    print('ACT-A:',act['status'],'\nACT-B:',act['status'],'\nACT physical DEV35:', 'AVAILABLE' if act.get('act_physical_results_available') else 'NOT RUN / UNAVAILABLE; not a synthetic0/35 result')
    print('------------------------------------------------------------\nMAIN FIGURE:',PAPER/'Fig_Main_SingleVariable_AB.png','\nTABLES:',PAPER/'TABLE_1_RETARGETING_FEASIBILITY.md',PAPER/'TABLE_2_REFERENCE_DEV35_PHYSICAL.md','\nREFERENCE VIDEOS (explicit failure cards where non-executable):')
    for name,item in sorted(replays['products'].items()):print(name,item['artifact']['path'])
    print('ACT VIDEOS:', 'AVAILABLE' if act.get('act_physical_results_available') else 'UNAVAILABLE','\nFINAL REPORT:',REPORT)
    print('FINAL_PAPER_RESULTS_READY')

if __name__=='__main__':main()
