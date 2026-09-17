"""Truthful prototype-only handoff; never converts nonexecution into failures."""
import csv
import io
from pathlib import Path
import numpy as np
from .io import ROOT, read, record, atomic_json, atomic_text


def csv_file(path, rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(path,stream.getvalue())


def table_md(rows):
    columns=list(rows[0])
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(r[c]) for c in columns)+' |' for r in rows)+'\n'


def run(out,resume=False):
    selection=read(out/'bootstrap/SELECTION.json');prototype=read(out/'prototype/RESULT.json')
    control=read(out/'common_control/RESULT.json')
    source_id=selection['prototype_source_id'];root=out/'prototype'/source_id
    coupled=read(root/'coupled_pose_region_v3/RESULT.json')
    region=read(root/'contact_region_v2/RESULT.json')
    assert not any(r['physically_run'] for r in prototype['results'])
    ledger=[]
    stages=['LEFT_GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','TRANSPORT','BIN_ENTRY','BIN_SETTLE','FULL_TASK']
    for i,sid in enumerate(selection['dev_source_ids']):
        methods=['WRIST_REFERENCE','INTERACTION_OURS'] if i%2==0 else ['INTERACTION_OURS','WRIST_REFERENCE']
        for method in methods:
            ledger.append(dict(schedule_order=len(ledger),dev_position=i,source_id=sid,method=method,
                planned=True,protocol_frozen=False,attempted=False,executable=False,physically_run=False,success='',
                terminal_class='NOT_ATTEMPTED_UPSTREAM',first_failure='M2_NOT_DEMONSTRATED',
                **{s:'NOT_ATTEMPTED' for s in stages}))
    for i,sid in zip(selection['ablation_dev_positions'],selection['ablation_source_ids']):
        ledger.append(dict(schedule_order=len(ledger),dev_position=i,source_id=sid,method='OURS_NO_COUPLING',
            planned=True,protocol_frozen=False,attempted=False,executable=False,physically_run=False,success='',
            terminal_class='NOT_ATTEMPTED_UPSTREAM',first_failure='M2_NOT_DEMONSTRATED',**{s:'NOT_ATTEMPTED' for s in stages}))
    assert len(ledger)==80
    csv_file(out/'PER_INSTANCE_RESULTS.csv',ledger)
    atomic_json(out/'SCHEDULE.json',dict(status='PREDECLARED_NOT_STARTED_NOT_FROZEN',rows=ledger,selection=record(out/'bootstrap/SELECTION.json')))
    main=[]
    for method in ['WRIST_REFERENCE','INTERACTION_OURS']:
        main.append(dict(method=method,planned=35,attempts=0,executable=0,physically_run=0,successes=0,
            infrastructure_unknown=0,not_attempted_upstream=35,plan_construction_rate='NOT_MEASURED',
            cumulative_scheduled_completions='0/35 (upstream stop; no evaluation)',
            observed_physical_completion='NOT_MEASURED',outcome_bounds='0..35 unevaluated successes; not a CI'))
    csv_file(out/'TABLE_MAIN_WRIST_VS_OURS.csv',main)
    atomic_text(out/'TABLE_MAIN_WRIST_VS_OURS.md','# Planned DEV35 comparison — not executed\n\n'+table_md(main)+'\nDEV35 is development data. No frozen-method comparison or physical success-rate estimate exists. Counts are scheduled-instance accounting, not evidence of 0% physical success.\n')
    ablation=[dict(dev_position=i,source_id=sid,ours='NOT_ATTEMPTED_UPSTREAM',ours_no_coupling='NOT_ATTEMPTED_UPSTREAM',paired_physical_difference='NOT_MEASURED')
              for i,sid in zip(selection['ablation_dev_positions'],selection['ablation_source_ids'])]
    csv_file(out/'TABLE_COUPLING_ABLATION_PAIRED10.csv',ablation)
    atomic_text(out/'TABLE_COUPLING_ABLATION_PAIRED10.md','# Predeclared paired-10 — not executed\n\n'+table_md(ablation)+
        '\nPlanned paired N=10; observed paired N=0. Positions are round(linspace(0,34,10)); all unique. No subset was chosen from successes. The separate TRAIN kinematic diagnostics below are not a paired-10 physical ablation.\n')
    diagnostics=[]
    for enabled,rows in coupled['results'].items():
        for row in rows:
            diagnostics.append(dict(source_id=source_id,enable_coupling=enabled,seed=row['seed'],
                overlap_position_disagreement_mm=1000*row['max_overlap_position_disagreement_m'],
                overlap_orientation_disagreement_rad=row['max_overlap_rotation_disagreement_rad'],
                modeled_contact_records=len(row['modeled_collision_records']),
                maximum_modeled_penetration_mm=1000*max((r['penetration_m'] for r in row['modeled_collision_records']),default=0),
                workspace_valid=row['workspace_valid'],overlap_consistent=row['overlap_consistent'],
                executable=False,physically_run=False,runtime_s=row['runtime_s'],nfev=row['nfev'],residual_calls=row['residual_calls']))
    csv_file(out/'TRAIN_COUPLING_DIAGNOSTICS.csv',diagnostics)
    atomic_json(out/'NUMERIC_SUMMARY.json',dict(main=main,planned_instances=80,attempted_dev_instances=0,
        prototype_source=source_id,prototype_training_sources=selection['train_source_ids'],
        prototype_search_versions=3,physical_method_rollouts=0,paired_physical_N=0,
        coupling_diagnostics=diagnostics,control=control,
        terminal_status='HYBRID_RETARGETING_PROTOTYPE_REPORT_ONLY'))
    parity='''# Method parity and current limits

WRIST_REFERENCE versus INTERACTION_OURS is a system/representation comparison. Ours versus the same code with `enable_coupling=False` is the proposed coupling ablation.

The phase adapter shares exact source recordings, timestamps, calibrated functional tool mapping, scene registration, G1 named limits and natural arm q0. Wrist keeps the registered source wrist/TCP spatial prior. Source phases are shared. Target fidelity is diagnostic; this prototype did not admit a complete low-fidelity physical path for either condition.

The source-only candidate selector shares unary scores, candidate banks and all non-coupling settings. On this prototype both toggles choose the same pair and the added factor is redundant at that selected fixed shared X. We make no attribution from that version.

The final shared-pose diagnostic uses the same 42 variables, three seeds, per-seed 120 function-evaluation limit, source unary weights, table region, fixed G_L/G_R, kinematics, interpolation checks and collision checks. Only cross-hand residuals are zeroed with the switch. Both conditions run every seed; cache accounting is shared. Residual-call counts and actual times are saved. Coupling changes predicted overlap consistency in this diagnostic, but every candidate is inadmissible. This is not physical attribution.

The geometry translation bank uses identical candidates and kinematic/collision admission for both toggles. No historical successful world pose, episode-specific IK repair, method-only arm correction, atlas, or ACT checkpoint is used. Raw numerical states remain untouched.

The shared planner takes `g1, goals, q0, config`; it has no method ID. Unit tests check source sensitivity, roles, common SE(3) equivariance, no independent hand rebasing, parity, coupling redundancy/effect and retiming derivatives. Tests do not establish source contact truth or physical success.

Still unqualified: contact patch transfer from ALOHA jaws to Dex3, collision correspondence with runtime USD, complete carried-object paths, full retiming integration, bounded event waits and source-conditioned natural-start physics. No controller/physics/evaluator freeze or full-method parity claim exists yet.
'''
    atomic_text(out/'METHOD_PARITY.md',parity)
    contract='''# Hybrid method contract — prototype, not an evaluation freeze

Task: existing rigid plush-surrogate doll, left acquisition/lift, bimanual handoff, right transport, bin entry and settle. ACT/VLA and real hardware are excluded.

T_XY maps Y into X. G_L=T_LO and G_R=T_RO are object-in-hand transforms. During dual support the proposed model requires FK_L(q_L)G_L ≈ FK_R(q_R)G_R ≈ X across the overlap. This is predicted planning geometry only; the PhysX object is dynamic and contact-driven.

Source initial XY is reused from per-recording image registration. Yaw uses the existing PCA rule modulo 180 degrees. Z is explicitly inferred from table height and the existing visual envelope. The left contact relation is fitted only before lift. Receiver relation is inferred from giver rigid carry during the recorded overlap. No observed source force, dynamic object track or certified source contact patch exists. Closing intent is not ownership. Absolute free-space object orientation is a source prior; fixed relative contact frames preserve object-relative axes. This choice is an unqualified development model, not a demonstrated grasp capability.

Stage definitions for any future freeze must use measured sustained contact, loss of table support, retained object rise, receiver contact followed by controlled giver release and right-only retention, measured transport, valid-bin-region release and at least 1 second settle. Existing scoring tolerances are preserved. No-acquisition must be NO_ACQUISITION, and premature-in-bin release must be distinguished from clean release. No source label establishes a measured stage.

Current physical evidence is common calibration only. The current scorer reports its exact legacy release classification separately. No new method threshold has been frozen. Raw joint excursions, contacts and object states are preserved; no attachment, hidden force or post-initialization object pose write is permitted.

All planned DEV instances use matched scenes and natural open-hand q0, one candidate under a common fixed total budget and at most one primary physical attempt. None has begun. TRAIN is calibration/development; DEV35 is development evaluation, not untouched test. Planned ablation positions are 0,4,8,11,15,19,23,26,30,34; reuse corresponding Ours outcomes if that stage is eventually reached.

Terminal classes: SOURCE_EVIDENCE_MISSING, NO_PLAN_WITHIN_FIXED_BUDGET, VALID_PLAN_TASK_SUCCESS, VALID_PLAN_TASK_FAILURE, EXECUTION_ABORT_PHYSICAL_VALIDITY, INFRASTRUCTURE_INVALID, NOT_ATTEMPTED_UPSTREAM. Unknown is separate from failure. Zero physical rollouts means NOT MEASURED. Source recording is the paired unit; seed candidates are not independent episodes.

Quintic retiming is implemented and derivative-tested. Existing 28-joint velocity ceilings are authoritative controller bounds; the recovered per-joint acceleration envelopes are empirical calibration values, not manufacturer ratings. Complete collision/continuity rechecking after interpolation and retiming remains unqualified, so no executable command was exported.
'''
    atomic_text(out/'METHOD_CONTRACT.md',contract)
    reused=read(out/'bootstrap/REUSE_COMPONENTS.json')
    modified=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tools/contact_coordination').glob('*.py') if p.name not in {'__init__.py','bootstrap.py','common_control.py','io.py','run_study.py'})
    modified+=['configs/contact_coordination/hybrid_development_v1.json']
    controls='\n'.join(f"- {r.get('control')}: {r['status']}; evidence `{r['run']}`." for r in control['results'])
    report=f'''# Hybrid source-conditioned retargeting: prototype evidence

TASK: existing doll handoff-and-bin
SOURCE_CONDITIONED_FULL_TASK: NOT_DEMONSTRATED
WRIST_DEV35: 0 attempts / 0 executable / 0 physically run / 0 successes / 0 infrastructure unknown; 35 NOT_ATTEMPTED_UPSTREAM
OURS_DEV35: 0 attempts / 0 executable / 0 physically run / 0 successes / 0 infrastructure unknown; 35 NOT_ATTEMPTED_UPSTREAM
COUPLING_ABLATION: planned paired N=10 / observed paired N=0 / physical difference NOT_MEASURED; kinematic factors meaningful only in final TRAIN diagnostic
ACT_VLA_TRAINING: NOT_RUN_BY_DESIGN
PHYSICAL_STUDY: NOT_EXECUTED
REPORT_PACKAGE: PARTIAL (prototype evidence package complete; physical paper package unavailable)
TERMINAL_STATUS: HYBRID_RETARGETING_PROTOTYPE_REPORT_ONLY

## Evidence and stop reason

The deterministic prototype is TRAIN recording `{source_id}`. Additional pilot recordings were predeclared as `{selection['train_source_ids'][1]}` and `{selection['train_source_ids'][2]}`. All three source phase records were built, but pilot physics was not started because the prototype never produced a complete admissible task.

Three bounded strategies were retained: source-only phase/keyframe targets; a three-candidate geometry-derived handoff translation bank; and a joint fit of both hands with the shared object pose implicit in FK(q)G. The first two could not realize the fixed giver contact orientation/pose at handoff. The third produced consistent predicted object poses for two coupled seeds, but all six coupled/independent candidates have modeled hand collisions and fail the declared workspace checks. No invalid candidate was passed to physics.

For fixed seed 0, coupling enabled gives {diagnostics[3]['overlap_position_disagreement_mm'] if diagnostics[3]['enable_coupling']=='True' else next(r['overlap_position_disagreement_mm'] for r in diagnostics if r['enable_coupling']=='True' and r['seed']==0):.3f} mm maximum predicted object-position disagreement across 21 interpolated overlap states. The disabled result is {next(r['overlap_position_disagreement_mm'] for r in diagnostics if r['enable_coupling']=='False' and r['seed']==0):.3f} mm. Every seed is reported in `TRAIN_COUPLING_DIAGNOSTICS.csv`; seed 0 is the fixed illustration rule, not a selected successful run. This does not measure physical handoff improvement.

The smallest unresolved engineering issue is a supported Dex3 paired-contact realization that preserves source grasp/closing meaning without modeled hand-to-hand penetration. Source jaw-relative contacts were inferred but do not certify a feasible pair of larger Dex3 hands. Complete carried-object connections, runtime collision correspondence and natural-start event integration remain outstanding. Bounded failures do not prove global impossibility. No source pose was replaced with a successful scripted world pose.

## Common physical evidence

{controls}

Two hash-compatible standalone controls were independently rescored. One new full scripted calibration replay was completed after a 360-second infrastructure timeout of an earlier invocation; the interrupted outcome remains UNKNOWN and is not a physical failure. The calibration command starts open but with task-specific arm placement, so it is not a natural-start Ours task. See `common_control/RESULT.json` and raw traces for exact contact, release, numerical and retention findings.

Current M1 status: `{control['status']}`. Mechanical capability and full milestone qualification are separate: runtime/offline collider equivalence and a natural-start approach remain unqualified. M2 not demonstrated; M3 partial offline interface tests only; M4 not started; M5 prototype handoff only.

## Denominators and claims

All 80 intended DEV method instances remain in `PER_INSTANCE_RESULTS.csv` as NOT_ATTEMPTED_UPSTREAM. No study freeze exists, no A/B sample was evaluated, and observed physical TSR is NOT MEASURED. The zero completed scheduled instances do not imply 0% physical TSR. There are 35 unevaluated primary instances per method and 10 unevaluated ablation instances. No binomial confidence interval or paired hypothesis test is applicable. Potential outcomes remain completely unobserved; paired-10 would be exploratory if eventually run. No desired success target was used as a tuning or acceptance threshold.

The rigid surrogate is not validated soft-body physics. The controller uses privileged simulated state. There is no real-robot, VLA or policy-training claim. Segmentation, retiming and planning alone are not claimed as novel.

## Reproduction and remaining work

Use the offline interpreter `/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python` and the simulation interpreter `/home/jbnu/miniconda3/envs/isaaclab6/bin/python`. Exact inputs, source IDs and assets are in `REUSE_MAP.md`, `bootstrap/REUSE_COMPONENTS.json`, phase records and per-run dependency manifests. Existing uncommitted changes were preserved in `bootstrap/`.

Run `/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python tools/contact_coordination/run_hybrid_study.py --run-dir {out} --stage report --resume` to reproduce this report. Prototype, control-audit, report and diagnostic-replay stages are implemented. Pilot, freeze and DEV execution stages are guarded and remain unimplemented/unqualified; this is not a completed end-to-end converter. Development budgets are not an evaluation freeze. The immediate next task is contact-pair calibration/admission using the existing primitives and exact runtime colliders, then a complete source-conditioned prototype. No historical dense TRAIN11 recovery or new unrelated project is warranted.

REUSED_COMPONENTS:
'''+ '\n'.join(f"- {r['component']}: `{r['file']['path']}`" for r in reused)+ '\n\nMODIFIED_COMPONENTS (new isolated adapters/config; prior tracked edits preserved):\n'+ '\n'.join(f'- `{p}`' for p in modified)+'\n'
    atomic_text(out/'FINAL_REPORT.md',report)
    atomic_text(out/'METHODS_DRAFT_KO.md',f'''# 방법 초안 — 구현 및 검증 범위

기존 ALOHA 인형 전달·상자 넣기 데이터, FK, 기능적 TCP 보정, 물체 초기 위치 등록, G1/Dex3 모델, 접촉 제어기와 PhysX 평가기를 재사용하였다. 학습과 실제 하드웨어 실행은 수행하지 않았다. TRAIN40은 보정·개발 자료이며 DEV35는 미관측 테스트 세트가 아닌 개발 평가 자료이다.

밀집 손목 궤적은 공간 사전정보로 남기고 접근, 획득, 들어올림, 전달 중 세 시점, 오른손 운반, 배치 목표를 구성하였다. 초기 물체 XY는 해당 원본 녹화의 영상 등록 결과를 사용하였다. Z와 yaw, 들어올리기 전 왼손-물체 관계, 전달 중 오른손-물체 관계의 관측·추론 범위를 PHASE_RECORD.json에 명시하였다. 원본 힘·접촉·동적 물체 자세는 관측되지 않았다.

T_XY는 Y 좌표를 X로 변환한다. 전달 구간의 결합 항은 FK_L(q_L)G_L과 FK_R(q_R)G_R이 같은 물체 자세를 나타내도록 유도한다. 마지막 개발 진단에서는 동일한 세 초기값, 단항 비용, 관절 한계, 공간 영역, 반복 예산을 사용하고 enable_coupling만 바꾸었다. 밀집 보간된 겹침 구간도 검사하였다. 이 강체 가정은 오프라인 예측이며 물리 실행의 부착 조건이 아니다.

손목 기준법과 제안법 비교는 시스템/표현 비교이며 단일 항 제거 실험으로 부르지 않는다. 제안법과 결합 비활성화 조건만 결합 항 귀속 실험에 해당한다. 현재는 완전한 유효 경로 및 자연 시작 물리 실행이 구현·검증되지 않아 최종 비교 결과로 사용할 수 없다.

원본 단위 결정적 개발 대상: {source_id}. 추가 파일 경로와 재현 절차는 FINAL_REPORT.md와 REUSE_MAP.md를 따른다.
''')
    atomic_text(out/'RESULTS_AND_LIMITATIONS_KO.md','''# 결과 및 한계 초안

원본 조건부 전체 과제 성공은 입증하지 못하였다. 세 가지 제한된 TRAIN 개발 전략을 실행하고 실패한 후보를 모두 보존하였다. 첫 전략에서 접근·획득·들어올림의 SE(3) 목표는 수치적으로 실현되었으나 전달 목표는 실패하였다. 접촉 관계를 고정한 채 물체 자세를 자유 변수로 두면 결합 항이 예측된 전달 일관성을 개선하지만, 모든 후보에 모델링된 손 충돌과 공간 제약 위반이 남았다. 이 결과는 물리 성공률 또는 실제 전달 성능 개선의 증거가 아니다.

공통 제어의 실제 측정 상태, 접촉, 물체 운동은 보정 증거로 별도 저장하였다. 스크립트 보정 성공을 Ours 성공으로 집계하지 않았다. 초기 인프라 시간초과는 물리 실패가 아닌 미확인 결과이다. 물리 상태를 자르거나 임계값을 넓히지 않았다.

DEV35 손목 35회, Ours 35회 및 사전 선언된 결합 비활성화 10회는 모두 상위 단계 미완료로 실행되지 않았다. 관측된 물리 TSR은 0%가 아니라 NOT MEASURED이다. 표의 0/35는 예정된 파이프라인의 완료 건수일 뿐 물리 성능 추정치가 아니다. 관측된 대응 표본은 0개이므로 신뢰구간, 유의확률 및 방법 우월성을 제시하지 않는다.

주요 한계는 ALOHA의 추론된 파지 관계를 Dex3의 비관통 접촉 쌍으로 변환하는 능력의 미검증, 런타임 convex collider와 오프라인 충돌 형상의 불일치, 자연 시작 전체 경로 및 이벤트 대기의 미통합이다. 강체 인형 대용체는 연성 물리 검증을 의미하지 않으며, 시뮬레이터 상태를 사용하는 특권적 접근이 있다. 실제 로봇 및 VLA 결과를 주장하지 않는다.
''')
    atomic_text(out/'CURRENT_STATUS.md','HYBRID_RETARGETING_PROTOTYPE_REPORT_ONLY\n\nM1: '+control['status']+'; full collider/natural-start qualification incomplete.\nM2: NOT_DEMONSTRATED. M3: offline parity interface only. M4: NOT_STARTED. M5: prototype evidence/report; physical study package PARTIAL.\n\n80 planned DEV instances retained as NOT_ATTEMPTED_UPSTREAM. Physical method TSR NOT MEASURED. See FINAL_REPORT.md.\n')
    atomic_text(out/'CHATGPT_UPDATE.md',(out/'CURRENT_STATUS.md').read_text())
    return dict(status='PROTOTYPE_REPORT_COMPLETE',terminal_status='HYBRID_RETARGETING_PROTOTYPE_REPORT_ONLY',physical_study='NOT_EXECUTED',report_package='PARTIAL')
