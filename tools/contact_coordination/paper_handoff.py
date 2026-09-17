"""Truthful paper handoff for the verified TRAIN40 / required ACT study."""
from pathlib import Path
import subprocess
from .io import ROOT,read,record,atomic_json,atomic_text


def final_status(out,n):
    complete=n['ACT']['paired'].get('paired_N')==35 and n['reference_recorded']==20
    training=out/'act_training/ACT_TRAINING_COMPLETE.json'
    replays=out/'replays/CURRENT_REPLAY_STATUS.json'
    complete=complete and training.exists() and n['matched_counts'] is not None and n['matched_counts']['paired_usable']>0
    complete=complete and replays.exists() and read(replays).get('all_required_ACT_replays_verified',False)
    if complete:return 'HYBRID_ACT_A_B_DEV35_PAPER_RESULTS_READY'
    zero=out/'matched_datasets/ZERO_PAIRED_INTERFACE_AUDIT.json'
    if n['generation_recorded']==80 and n['reference_recorded']==20 and zero.exists():
        audit=read(zero)
        if audit['status']=='ZERO_PAIRED_SUPERVISION_AFTER_COMMON_INTERFACE_AUDIT' and audit['counts']['paired_usable']==0:
            return 'HYBRID_ACT_A_B_EXPERIMENT_PREREQUISITE_UNRESOLVED'
    return 'ACTIVE_EXPERIMENT_CONTINUE_AVAILABLE_STAGES'


def write_report(out):
    n=read(out/'PAPER_NUMERIC_SUMMARY.json')
    for dep in n['evidence']:
        if record(dep['path'])!=dep:raise ValueError('Numeric inputs changed; rerun analysis')
    milestone=read(out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json')
    for key in ('score','trace','video','alignment'):
        if record(milestone[key]['path'])!=milestone[key]:raise ValueError('Changed prototype evidence')
    score=read(milestone['score']['path'])
    if not score['physical_validity'] or not score['stages']['FULL_TASK']:raise ValueError('Unverified source full task')
    split=read(out/'SPLIT_CONTRACT.json');pilot=read(out/'TRAIN_pilot/PILOT_LEDGER.json')
    metrics=read(out/'MECHANISTIC_METRICS.json') if (out/'MECHANISTIC_METRICS.json').exists() else None
    counts=n['matched_counts'];paired=counts['paired_usable'] if counts else None
    frozen=read(out/'generation_freeze/CONTRACT.json')
    prototype_frames=int(milestone['frames']);prototype_seconds=prototype_frames/float(milestone['fps'])
    diagnostic_path=out/'dataset_interface_diagnostic/RESULT.json'
    diagnostic_frames=read(diagnostic_path)['frames'] if diagnostic_path.exists() else 'unavailable'
    pilot_counts={c:dict(plans=sum(r.get('full_task_plan') is True for r in pilot['rows'] if r['condition']==c),
        physical=sum(r.get('physical_run') is True for r in pilot['rows'] if r['condition']==c),
        valid=sum(r.get('physical_validity') is True for r in pilot['rows'] if r['condition']==c),
        successes=sum(r.get('task_success') is True for r in pilot['rows'] if r['condition']==c)) for c in ('A','B')}
    state=final_status(out,n);a=n['ACT']['conditions']['A'];b=n['ACT']['conditions']['B']
    actual_training={};lineage=[]
    for c in ('A','B'):
        selected=out/'act_training'/c/'SELECTED_CHECKPOINT.json'
        if selected.exists():
            s=read(selected);actual_training[c]=s['checkpoint'];lineage.append(record(selected))
        else:actual_training[c]='NOT AVAILABLE — no compatible selected target-domain checkpoint'
    def success(c):
        x=n['ACT']['conditions'][c]
        if x['full_task_rate'] is not None:return f"{x['full_task_successes']}/35 = {100*x['full_task_rate']:.1f}%"
        if x['unresolved_rate_bounds'] is not None:return f"UNRESOLVED, rate bounds {x['unresolved_rate_bounds']}"
        return 'NOT MEASURED'
    effect=n['ACT']['paired'].get('difference_pp');effect_text='NOT ESTIMABLE' if effect is None else f'{effect:.1f} percentage points'
    reference=n['reference'];rs=n['reference_paired']
    reference_statistics=(f"Reference pipeline completion is {reference['B']['full_task_successes']}/10 for Ours and {reference['B_NO_COUPLING']['full_task_successes']}/10 for coupling off. The paired pipeline difference is {rs['difference_pp']:.1f} percentage points, with conservative exact95% interval [{rs['paired_interval95_pp'][0]:.2f}, {rs['paired_interval95_pp'][1]:.2f}] and exact McNemar p={rs['McNemar_exact_p']:.3g}. These statistics count no-plan cases as pipeline noncompletion; they do not estimate conditional physical success or establish physical equivalence."
        if rs.get('paired_N')==10 else 'Paired reference pipeline uncertainty is not estimable until the declared outcomes are resolved.')
    result_lines=f'''TRAIN SOURCES: 40
A VALID TARGET DEMONSTRATIONS: {n['converters']['A']['complete_valid_supervision']}
B VALID TARGET DEMONSTRATIONS: {n['converters']['B']['complete_valid_supervision']}
PAIRED ACT TRAINING SOURCES: {paired if paired is not None else 'PENDING ALL80 CONVERSION OUTCOMES'}
TRAIN_EVAL_SOURCE_OVERLAP: {split['train_eval_source_overlap']}
ACT-A CHECKPOINT: {actual_training['A']}
ACT-B CHECKPOINT: {actual_training['B']}
ACT-A DEV35: 35 scheduled / {a['valid_physical_rollouts']} valid physical / {a['full_task_successes']} observed full-task successes / {a['infrastructure_unknown']} infrastructure unknown / {a['not_attempted']} not attempted
ACT-B DEV35: 35 scheduled / {b['valid_physical_rollouts']} valid physical / {b['full_task_successes']} observed full-task successes / {b['infrastructure_unknown']} infrastructure unknown / {b['not_attempted']} not attempted
ACT-A FULL TASK: {success('A')}
ACT-B FULL TASK: {success('B')}
B-A: {effect_text}
REFERENCE COUPLING ABLATION: {n['reference_recorded']}/20 recorded reference attempts; {rs.get('paired_physical_N',0)} paired valid physical sources; separate from ACT
COMMON_CONTROL: VERIFIED within bilateral component-control scope
SOURCE_CONDITIONED_FULL_TASK: DEMONSTRATED on the fixed TRAIN development source
ACT_RUNTIME_MODE: {'Selected target-domain closed-loop ACT verified; see ACT_policy/SANITY.json' if (out/'ACT_policy/SANITY.json').exists() else 'NEW TARGET-DOMAIN POLICY PAIR NOT VERIFIED; legacy inference is provenance only'}
PHYSICAL_STUDY: {'COMPLETE' if state.endswith('PAPER_RESULTS_READY') else 'PARTIAL (converter development/recorded reference evidence; required ACT comparison unavailable or incomplete)'}
REPORT_PACKAGE: {'COMPLETE' if state.endswith('PAPER_RESULTS_READY') else 'PARTIAL relative to the requested ACT paper; verified prerequisite evidence reported'}
TERMINAL_STATUS: {state}
'''
    shortfall=('All80 frozen TRAIN conversion attempts are recorded. Their complete valid supervision intersection is zero. '
               f'The separate real{diagnostic_frames}-frame G1 dataset interface diagnostic passed, and the common-interface audit distinguishes missing/invalid motion from serialization failures. '
               'There is no authorized nonempty paired dataset to train ACT-A/B. Repeating the development B success, copying it as A, using legacy source images, or duplicating sources would change the experiment. '
               'The exact missing prerequisite is a nonempty set of source IDs with complete valid A and B dynamic target-domain supervision under a common protocol.'
               if paired==0 and n['generation_recorded']==80 else
               'Read the exact matched counts and stage receipts. Pending ACT training/interface/evaluation steps must continue when a nonempty paired dataset exists; reference evidence cannot close the required ACT experiment.')
    proto=(metrics['rows'][0] if metrics else None)
    mechanism=(f"The demonstrated task lasts {proto['actual_execution_time_s']:.3f}s versus {proto['source_duration_s']:.3f}s of source recording. Its selected handoff object center moves {proto['handoff_relocation_from_inferred_source_prior']['position_mm']:.3f}mm from the source-inferred handoff prior. The median measured dual-contact predicted object disagreement is {proto['measured_contact_relation_statistics']['dual']['position_mm']['median']:.3f}mm over {proto['measured_contact_relation_statistics']['dual']['sampled_control_frames']} sampled control frames. These are one development illustration, not population estimates. Raw source-wrist deviations, orientation changes, target-contact residuals and timing are in MECHANISTIC_METRICS.json; no dense tracking fidelity claim is made." if proto else 'Detailed mechanistic diagnostics remain in the saved goals, command/measurement traces and contact-selection records.')
    report=f'''# Required ACT-A/B study: verified evidence and remaining prerequisites

```text
{result_lines}```
## What was demonstrated

The fixed source `GoPark_20260820_152058` now completes natural approach, actual left acquisition/lift, left-to-right handoff, right-only retention, right transport, bin entry, clean release and settling in dynamic G1/Dex3 simulation. The first independently verified current-run milestone contains {prototype_frames}control frames over {prototype_seconds:.3f}seconds. Its exact version, source, plan, trace and replay are linked in FIRST_SOURCE_CONDITIONED_FULL_TASK.json. The independent score, full measured articulation, object/contact trace and ordinary replay agree. All preceding development versions and failures remain preserved; their attempt numbers are not pooled across historical runs. No object attachment, hidden carrying force or post-initialization object transport write was used. This is a source-conditioned converter milestone, not an ACT rollout and not a frozen TRAIN40 result.

{mechanism}

The current-compatible left/right component controls were reused after every dependency hash and trace hash was checked. Their scope is acquisition, lift, retention and natural release in the checked configurations. The rigid20g plush surrogate and150mm bin use the existing qualified parameters. This does not validate soft-body physics, every source pose, or all PhysX cooking/filter configurations.

## Frozen conversion and supervision

The same fixed five-source A/B pilot recorded10 attempts. A produced {pilot_counts['A']['plans']}/5 complete plans, {pilot_counts['A']['physical']} physical runs and {pilot_counts['A']['successes']} full successes. B produced {pilot_counts['B']['plans']}/5 complete plans, {pilot_counts['B']['physical']} physical runs, {pilot_counts['B']['valid']} valid physical runs and {pilot_counts['B']['successes']} full successes. The pilot was an integration check, not a required5/5 score. General task-frame normalization, bounded object-relative candidate directions, explicit contact modes/carried-object checks, retained connecting knots and passive-receiver preparation were repaired before freeze. The verified generalization gate and exact decisions are in GENERALIZATION_REPAIR_READY.json and GENERALIZATION_REPAIR_DECISIONS.md. The generation contract then fixed {len(frozen['dependencies'])} dependencies,80 paired source/method instances,900seconds total planning per instance and at most one primary physical attempt.

The recorded converter totals are in TABLE_CONVERTER_RESULTS.csv and all80 scheduled instances in TRAIN40_PER_INSTANCE_RESULTS.csv. Planning failure does not create a grasp observation. The raw schedule field `physical_primary_attempts=1` is the predeclared maximum permission, not an execution count; reported physical counts use `physical_run`, process receipts and measured traces. Complete valid task failures can supply ACT supervision but are explicitly non-expert; incomplete, invalid and diagnostic traces cannot. Source membership for learning is the identical ordered usable intersection, without duplication or method-specific filtering.

{shortfall}

## Required policy result and separate attribution

ACT-A Full Task Success is {success('A')}; ACT-B is {success('B')}; B−A is {effect_text}. TABLE_ACT_DEV35_RESULTS.csv and PER_EPISODE_ACT_RESULTS.csv preserve all35 scheduled sources per policy. Unexecuted cases are NOT_ATTEMPTED_UPSTREAM and physical TSR is NOT MEASURED, not0%. Policy-caused failure/abort would remain a resolved failure; common infrastructure invalidity remains unknown. Exact paired McNemar and conservative paired confidence intervals are produced only for resolved matched policy outcomes.

Legacy ACT-A/B checkpoints performed genuine inference in earlier diagnostics, but their effective source-image/lagged-state supervision differs from the required G1 RGB/current measured-state interface. They are not reused as the new policies. Both new models must use the same authorized paired source membership and common100000step fixed checkpoint rule, without DEV35 success selection. The planned primary runtime consumes current G1 RGB and measured state, outputs50-action chunks, and uses the same safety/servo adapter. Source event clocks, teacher trajectories and the demonstration controller are absent from policy spatial control.

The paired10 coupling branch contains {n['reference_recorded']}/20 reference attempts. Ours has {reference['B']['complete_valid_plans']}/10 complete plans and {reference['B']['valid_physical_rollouts']} valid physical runs; coupling-disabled Ours has {reference['B_NO_COUPLING']['complete_valid_plans']}/10 and {reference['B_NO_COUPLING']['valid_physical_rollouts']} respectively. Its pipeline and physical denominators are separate. Predicted handoff fits, where completed, are saved in REFERENCE_COUPLING_CANDIDATE_GEOMETRY.csv. Algebraic nonredundancy and geometric differences do not alone prove a physical benefit, and this reference experiment cannot establish a learned-policy coupling effect.

{reference_statistics}

## Limits and exact continuation

DEV35 is development evaluation, not untouched testing. Source initial registration, pre-lift tool/object relations, hand roles and annotated/inferred event windows condition the converter; source forces and dynamic object tracking are not invented. Absolute source handoff orientation is partly unknown. Qualified target contact frames and TRAIN-calibrated gravity/load predictions permit robot morphology adaptation; raw spatial deviations remain reported. No independent left/right rebasing or task-specific source-ID world-pose branch is used.

The earliest frozen failure per instance and raw physical-validity excursions are retained. A finite failed search is not global infeasibility. No tolerance was widened to meet a requested score. A new converter revision to address systematic acquisition/receiver/path failures would require a new generation protocol and symmetric A/B attempts, followed by real matched data, both ACT trainings, policy sanity and70 fresh policy trials. Existing frozen outcomes must remain separate. The next engineering target is complete valid common source coverage, not a desired A/B score.

The saved Wrist failure analysis also separates joint IK from grasp-model compatibility. Read TRAIN40_conversion/diagnostics/A_FROZEN_FIRST_FAILURE_INTERPRETATION.md and A_FROZEN_CARRY_MODEL_COMPATIBILITY.json: the shared calibrated pickup transform can predict an invalid carried-object pose at an otherwise solved Wrist acquisition configuration. These are model-based collision rejections, not measured failed lifts. They limit interpretation of Wrist plan construction and identify a future common modeling/search issue; they do not establish that Wrist transfer is globally infeasible or that ACT-B is superior.

REPRODUCE.md documents actual stage/resume commands and their current validation limits. REUSE_MAP.md gives exact modules, assets and compatibility evidence. generation_freeze/CONTRACT.json, bootstrap/ENVIRONMENT_VERSIONS.json, source/manifests and per-attempt commands/hashes preserve lineage. FIRST_SOURCE_CONDITIONED_FULL_TASK.json and replays/FIRST_SOURCE_CONDITIONED_FULL_TASK.mp4 provide the actual development full-task evidence. Required ACT mosaics exist only if actual policy traces have been rendered and verified; cards cannot substitute for policy physics.

No real hardware, VLA integration, new source collection, second task or soft-body claim is included. Segmentation, common planning and retiming alone are not claimed as novel. The intended contribution is source interaction constraint conversion; downstream ACT usefulness requires the missing or completed actual paired policy evidence above.
'''
    atomic_text(out/'FINAL_PAPER_READY_REPORT.md',report);atomic_text(out/'FINAL_REPORT.md',report)
    atomic_text(out/'METHODS_DRAFT_KO.md',f'''# 방법 초안

기존 인형 왼손 파지·양손 전달·오른손 운반·상자 배치 과제와 ALOHA 원본을 재사용하였다. 승인된 분할은 TRAIN40과 서로 겹치지 않는 DEV35이며, DEV35는 반복적으로 검토된 개발 평가 자료이다.50개 학습 자료를 주장하지 않는다.

A는 기능적 TCP를 정확히 보정한 손목 공간 사전정보를 사용한다. B는 동일 원본의 단계·접촉 관계를 G1/Dex3의 보정된 접촉 후보로 변환하고 양손이 같은 물체 자세를 나타내도록 결합한다. T_AB는 B에서 A로의 변환이며, 목표 손 자세는 T_WH = X inverse(G)이다. 고정 도구 변환은 한 번만 적용한다. 두 조건은 장면, IK/연결, 충돌 검사, 시간 조정, 접촉 제어와 물리 설정을 공유한다. 밀집 손목 오차는 진단이며 물리 실행의10mm 게이트가 아니다.

결합 제거 조건은 동일 코드에서 교차 손 물체 위치·회전 잔차만 비활성화한다. 단항 비용, 후보, 초기값, 공간 영역과 예산은 공유한다. 물리적 안전 검사에서 결합 후보를 몰래 다시 선택하지 않는다. 강체 파지 가정은 계획 형상 예측이며 런타임 부착이 아니다.

고정 TRAIN 원본의 전체 과제를 실제 동적 측정으로 확인한 뒤5개 추가 원본의 A/B 파일럿을 수행하였다. 이후40×2회 변환을 공통900초 계획 예산으로 고정하였다. 완전하고 유효한 G1 RGB·측정28차원 상태·실제 실행28차원 명령이 있는 공통 원본 교집합만 ACT 학습에 사용한다. 관측은 해당 명령 이전이고30Hz로 정렬된다. 유효한 과제 실패는 비전문가 표지를 유지하며, 잘린·잘못된 물리 궤적이나 실패 계획에 가짜 명령을 추가하지 않는다.

현재 공통 학습 가능 원본 수: {paired if paired is not None else '집계 중'}. ACT는 필수 후속 실험이다. 기존 원본 영상 정책은 목표 G1 관측 계약과 달라 대체하지 않는다. 새 A/B는 같은 구조·시드·옵티마이저·100000스텝 최종 체크포인트 규칙을 사용한다. 정책 평가 시 현재 G1 영상과 측정 상태만 ACT에 입력하고 원본 궤적·이벤트 시계·시연 제어기는 공간 행동에 관여하지 않는다. 실제 학습 및 정책 검증 완료 여부는 FINAL_REPORT.md를 따른다.
''')
    atomic_text(out/'RESULTS_DRAFT_KO.md',f'''# 결과 초안 — 측정 범위 유지

고정 TRAIN 원본 GoPark_20260820_152058에서 자연 시작, 실제 파지·들기·전달·오른손 소유·운반·상자 진입·놓기·안정화를 입증하였다. 현재 실행의 최초 검증 이정표는 {prototype_seconds:.3f}초, {prototype_frames}제어 프레임의 실제 측정 시도이다. 앞선 개발 버전과 부분 단계 진단·실패 기록도 보존했으며, 서로 다른 실행 폴더의 시도 번호를 합쳐 전체 과제 실패 횟수로 해석하지 않는다. 고정 TRAIN5의 실행 가능한 계획은 A {pilot_counts['A']['plans']}/5, B {pilot_counts['B']['plans']}/5이고 물리적 전체 성공은 각각 {pilot_counts['A']['successes']}개와 {pilot_counts['B']['successes']}개이다. 이 결과는 기준 변환기의 개발 이정표이며 ACT 성공률이 아니다.

동결된 TRAIN40 A/B 변환 기록: {n['generation_recorded']}/80. A의 완전 유효 시연은 {n['converters']['A']['complete_valid_supervision']}개, B는 {n['converters']['B']['complete_valid_supervision']}개, 공통 ACT 원본은 {paired if paired is not None else '미확정'}개다. 인터페이스 진단의 단일 B 개발 시연을 A 자료로 복사하거나 학습 교집합에 추가하지 않았다.

주요 정책 결과: ACT-A {success('A')}, ACT-B {success('B')}, B−A {effect_text}. 미실행 결과는0/35 성공률로 보고하지 않는다. 대응 DEV35 전체 표와 단계 결과는 PER_EPISODE_ACT_RESULTS.csv에 있다. 실제 정책 결과가 없으면 대응 검정과 우월성 주장은 불가능하다.

별도 기준 수준 결합 실험은 {n['reference_recorded']}/20회 기록되었고 대응 유효 물리 표본은 {rs.get('paired_physical_N',0)}개다. 계획 가능성, 전체 예정 사례의 완료, 유효 물리 실행 조건부 완료를 구분한다. 이 실험은 ACT 학습 이후 결합 효과의 근거가 아니다.
''')
    atomic_text(out/'LIMITATIONS.md','''# Limits of the evidence

Successful source-conditioned development realizations after retained bounded TRAIN repairs do not estimate general reliability. The exact current pilot and frozen-generation counts are reported separately. Frozen source conversion can fail in acquisition, handoff connections, loaded geometry or physical contact despite a valid command plan. A weak pilot was not forced to a success quota. Finite search failures do not establish global impossibility.

Source contact forces, continuous object poses and some orientation requirements are unknown/inferred. The rigid20g plush surrogate is not validated soft-body physics. Privileged simulator state supports demonstration planning/control and common safety. Runtime/offline checks retain unresolved PhysX cooking and same-hand filter scope; numerical validity is not a universal solid-geometry certificate.

Target-domain ACT requires actual paired complete valid observation/action supervision. An isolated B development success and compatible tensor dimensions do not create that dataset. Missing ACT trials remain unmeasured. A single matched training-seed pair, if trained, is not multi-seed robustness. DEV35 is development evaluation, not untouched testing. Reference-level paired10 attribution is exploratory and cannot establish coupling effects after learning.

No real-robot, VLA, broad generalization or new-task result is claimed. Failure thresholds, raw measurements and old outcomes remain unchanged. Further method revisions must be versioned and evaluated symmetrically rather than combined with favorable earlier outcomes.
''')
    atomic_text(out/'NEXT_ACTIONS.md',f'''# Exact continuation from the recorded evidence

The missing prerequisite is nonempty, identical A/B source membership with complete valid target G1 observation/action supervision. The authorized TRAIN40 source data exist. The current matched count is {paired}. Do not substitute the successful development B trace, legacy source-image policies, partial trajectories or duplicated sources for the missing paired supervision.

Frozen first-failure counts are A: {n['converters']['A']['first_failures']}; B: {n['converters']['B']['first_failures']}. Read SAVED_REJECTED_PHASE_DETAILS.json for named goals, selected IK residuals/FK, finger modes and collision pairs/locations. It is a read-only extraction of retained bounded searches, not a new reachability experiment. Use the actual lowest-ID rejection and phase residual/geometry evidence in this run. Do not infer a particular collision or an IK defect from a historical report template. Future repairs must address demonstrated goal or connecting-path defects through the existing generator/backend; a workspace-radius check is not a reachability proof.

For this run, TRAIN40_conversion/diagnostics/A_FROZEN_FIRST_REJECTED_PHASES.json refines the outer acquisition-connection label, and A_FROZEN_CARRY_MODEL_COMPATIBILITY.json records the initial-object versus calibrated carry prediction. Check this compatibility before extending IK search. A different achieved grasp transform or geometry-aware acquisition search must preserve actual source/scene inputs and be applied by a common rule in a new symmetric protocol; do not conceal a copied B solution or waive table collision. The current diagnostic does not establish a source identity, cache or coordinate arithmetic bug.

Read FROZEN_PHYSICAL_VALIDITY_AUDIT.json and the linked PHASE_STOP/geometry/contact traces before changing receiver behavior. Only physically executed instances can establish acquisition, lift, receiver ownership or measured contact failure. No-plan Wrist cases have unattempted physical stages. The common sequence already permits a receiver candidate before controlled giver release and checks sole ownership afterward. Do not replace it with a circular ownership gate or a hidden runtime arm rescue. Inspect the first actual receiver-contact/clearance failure and preserve the current contact and validity thresholds.

Profile the retained TRAIN planning records for wall-budget failures before adding solver complexity. The development full-task success does not demonstrate that every frozen source can be planned within900seconds. Any materially changed candidate ordering, contact region, connection or timing algorithm requires a separately versioned common generation protocol and symmetric A/B attempts. Keep all80 outcomes of this version; do not tune from the later reference DEV outcomes or combine repaired B with old A. Use the same fixed TRAIN prototype and existing five-source pilot for the new version's integration checks. No new task, physics framework, hand atlas or ACT architecture is indicated by these records.

The real{diagnostic_frames}-frame G1 observation/action dataset diagnostic is separately recorded; check each frozen trace alignment receipt and infrastructure status before declaring its supervision usable. Reuse this verified interface. Once a nonempty common complete-valid source set exists, materialize both target-domain datasets, audit effective tensors, train both existing ACT implementations under the same100000-step selection rule, verify genuine current-observation policy control on TRAIN, and then run70 fresh paired DEV35 policy trials. The current ACT results remain unmeasured until those actions actually occur.
''')
    result=dict(status=state,numeric_summary=record(out/'PAPER_NUMERIC_SUMMARY.json'),
                report=record(out/'FINAL_PAPER_READY_REPORT.md'),selected_checkpoint_receipts=lineage,
                required_ACT_experiment_complete=state.endswith('PAPER_RESULTS_READY'),
                source_full_task_is_ACT_evidence=False,implementation=record(__file__))
    atomic_json(out/'PAPER_HANDOFF.json',result)
    return result


def verify(out):
    result=write_report(out)
    if result['status']=='ACTIVE_EXPERIMENT_CONTINUE_AVAILABLE_STAGES':
        raise ValueError('Required stages still available or active; no final terminal claim')
    n=read(out/'PAPER_NUMERIC_SUMMARY.json')
    assert n['generation_recorded']==80 and n['reference_recorded']==20
    initial=(out/'bootstrap/tracked_changes_initial.patch').read_bytes()
    if subprocess.check_output(['git','diff','--binary'],cwd=ROOT)!=initial:
        raise ValueError('Pre-existing tracked edits changed')
    figures=read(out/'figures/CURRENT_FIGURE_PROVENANCE.json')
    if figures['numeric_summary']!=record(out/'PAPER_NUMERIC_SUMMARY.json'):
        raise ValueError('Figures do not match the final numeric evidence')
    for item in figures['products']:
        if record(item['path'])!=item:raise ValueError('Changed figure')
    replay_status=read(out/'replays/CURRENT_REPLAY_STATUS.json')
    verified_replays=[]
    for replay in replay_status.get('TRAIN_failure_illustrations',[])+replay_status.get('reference_illustrations',[]):
        for key in ('video','trace'):
            if record(replay[key]['path'])!=replay[key]:raise ValueError('Changed measured replay evidence')
        if int(replay['decoded']['nb_read_frames'])!=replay['frames']:
            raise ValueError('Replay decode count differs from the saved physical frames')
        verified_replays.append(replay['video'])
    required=['FINAL_PAPER_READY_REPORT.md','METHODS_DRAFT_KO.md','RESULTS_DRAFT_KO.md','LIMITATIONS.md',
              'PAPER_FIGURE_DIRECTION.md','REPRODUCE.md','TABLE_CONVERTER_RESULTS.csv','TABLE_ACT_DEV35_RESULTS.csv',
              'TABLE_COUPLING_ABLATION.csv','PER_EPISODE_ACT_RESULTS.csv','MECHANISTIC_METRICS.json',
              'NEXT_ACTIONS.md','SAVED_REJECTED_PHASE_DETAILS.json','FROZEN_PHYSICAL_VALIDITY_AUDIT.json',
              'REUSE_MAP.md','METHOD_PARITY.md','DATASET_PARITY.md','STUDY_CONTRACT.md','CHECKPOINT_LINEAGE.md',
              'ACT_TRAINING_PARITY.md','ACT_POLICY_INTERFACE_SANITY.md','FINAL_TABLE_AUDIT.json','VISUAL_REVIEW.json',
              'SPLIT_CONTRACT.json','FIRST_SOURCE_CONDITIONED_FULL_TASK.json',
              'figures/CURRENT_FIGURE_PROVENANCE.json','replays/CURRENT_REPLAY_STATUS.json']
    result.update(final_files=[record(out/p) for p in required]+figures['products']+verified_replays,
                  verified_failure_replays=len(verified_replays),prior_tracked_edits_preserved=True,
                  verification='Outcome/trace/dependency checks, explicit ACT unavailability when applicable; file existence is not scientific completion.')
    atomic_json(out/'FINAL_VERIFICATION.json',result)
    return result
