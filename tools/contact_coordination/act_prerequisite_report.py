"""Evidence-backed partial ACT study package; unavailable outcomes stay absent."""
import csv,io,json,hashlib
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text

TERMINAL='HYBRID_RETARGETING_ACT_PREREQUISITES_UNRESOLVED'
STAGES=['LEFT_GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','RIGHT_TRANSPORT','BIN_ENTRY','BIN_SETTLE','FULL_TASK']


def csv_write(path,rows):
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);atomic_text(path,f.getvalue())


def analysis(out):
    split=read(out/'SPLIT_CONTRACT.json');selection=read(out/'bootstrap/SELECTION.json');sid=selection['prototype_source_id']
    short=out/'prototype'/sid/'morphology_acquisition_v4/physics_attempt_01'
    score=read(short/'INDEPENDENT_ACQUISITION_SCORE.json')
    interface=read(out/'ACT_interface_diagnostics/v1/INDEPENDENT_VERIFICATION.json')
    fit=Path(read(out/'target_repair/CURRENT_CONTACT_REGION_FIT.json')['path'])
    on,off=read(fit/'coupling_True.json'),read(fit/'coupling_False.json')
    geometry=read(out/'geometry_verification/RESULT.json')
    old=ROOT/'outputs/contact_coordination_hybrid/20260907T083735Z/prototype'/sid/'full_task_connection/ead64a3c8365/geometry_telemetry_diagnostic'
    compatibility=read(old/'DEPENDENCIES.json')
    assert all(record(d['path'])['sha256']==d['sha256'] for d in compatibility['files'])
    assert all(x['forbidden_count']==0 for x in geometry)
    assert split['authorized_training_count']==40 and not split['train_eval_source_overlap']
    assert selection['ablation_dev_positions']==np.round(np.linspace(0,34,10)).astype(int).tolist()
    n=dict(terminal=TERMINAL,task='existing doll handoff-and-bin',source_id=sid,
        common_control='VERIFIED_COMPONENT_CONTROLS',common_control_scope='Fresh bilateral acquisition/lift/retention/release; fresh natural-start source acquisition; hash-compatible archived receiver approach/giver release/right-only retention, with corrected measured geometry. Not a full-script reliability claim.',
        current_physical_executions=5,common_standalone_controls=2,source_acquisition_physical_runs=1,
        legacy_ACT_interface_physical_runs=2,source_conditioned_full_task='NOT_DEMONSTRATED',
        acquisition=score,interface=interface,geometry=geometry,archived_handoff_runtime_dependencies=record(old/'DEPENDENCIES.json'),
        requested_training_per_method=50,authorized_training_count=40,new_matched_training_per_method=0,
        selected_compatible_checkpoints=0,ACT_A_main_trials=0,ACT_B_main_trials=0,ACT_main_scheduled_intent_per_method=35,
        ACT_A_full_task_success='NOT_MEASURED',ACT_B_full_task_success='NOT_MEASURED',B_minus_A='NOT_ESTIMABLE',
        reference_coupling_paired_physical_N=0,reference_coupling_intended_paired_N=10,
        train_geometric_coupling=dict(source_episodes=1,seeds_per_condition=2,contact_candidates=1,
            on_position_disagreement_m=[x['position_disagreement_m'] for x in on],
            off_position_disagreement_m=[x['position_disagreement_m'] for x in off],
            coupled_admissible_stationary_goals=sum(x['valid_stationary_overlap'] for x in on),
            uncoupled_admissible_stationary_goals=sum(x['valid_stationary_overlap'] for x in off),
            physical_attribution=False,fit=record(fit/'CONFIG.json')),
        current_final_plan=read(out/'target_repair/TARGET_REPAIR_STATE.json')['latest'],
        freeze_complete=False,new_ACT_training_run=False,physical_study='NOT_EXECUTED',report_package='PARTIAL',
        unresolved=['No authorized disjoint50-source training split','No source-conditioned full task: PLACE remains collision-invalid or outside bounded IK tolerance','No matched dynamically realized G1 demonstrations or compatible trained ACT pair'])
    atomic_json(out/'NUMERIC_SUMMARY.json',n)
    ledger=[]
    for i,source in enumerate(split['evaluation_source_ids']):
        for method in ('ACT-A','ACT-B'):
            ledger.append(dict(paired_schedule_order=len(ledger),dev_position=i,source_id=source,policy=method,
                intended=1,protocol_frozen=0,physically_run=0,valid_physical_rollout=0,full_task_success='NOT_MEASURED',
                terminal='NOT_ATTEMPTED_UPSTREAM',unknown_common_infrastructure=0,
                **{s:'NOT_ATTEMPTED' for s in STAGES}))
    csv_write(out/'ACT_DEV35_PER_EPISODE_RESULTS.csv',ledger)
    main=[dict(policy=m,intended_cases=35,resolved_primary_outcomes=0,physical_runs=0,valid_primary_runs=0,
        successes='NOT_MEASURED',task_success_rate='NOT_MEASURED',not_attempted_upstream=35,
        common_infrastructure_unknown=0,paired_difference='NOT_ESTIMABLE',confidence_interval='NOT_APPLICABLE_NO_OBSERVATIONS') for m in ('ACT-A','ACT-B')]
    csv_write(out/'TABLE_ACT_A_VS_ACT_B_DEV35.csv',main)
    atomic_text(out/'TABLE_ACT_A_VS_ACT_B_DEV35.md','''# Primary ACT comparison — unavailable

| Policy | Intended DEV35 cases | Physical primary trials | Resolved outcomes | Full Task Success | Upstream not attempted |
|---|---:|---:|---:|---|---:|
| ACT-A | 35 | 0 | 0 | NOT MEASURED | 35 |
| ACT-B | 35 | 0 | 0 | NOT MEASURED | 35 |

B − A: NOT ESTIMABLE. No confidence interval or paired test applies. All counts from0 to35 remain logically possible per policy; this is not a confidence interval. No common infrastructure-invalid primary trial exists because none began. The planned70-row ledger is not70 failures. Two short legacy-checkpoint TRAIN interface diagnostics are excluded from this table. DEV35 is repeatedly inspected development evaluation, not untouched testing.
''')
    provenance=[dict(condition=m,requested_sources=50,authorized_disjoint_sources=40,source_pool=79,
        new_complete_dynamic_demos=0,new_ACT_training=0,legacy_ACT40_sources=40,legacy_state='LAGGED_COMMAND_SURROGATE',
        legacy_RGB='ALOHA_CAM_HIGH',compatible_selected_checkpoint=0) for m in ('WRIST_REFERENCE','INTERACTION_OURS')]
    csv_write(out/'TABLE_CONVERSION_AND_DATASET_PROVENANCE.csv',provenance)
    atomic_text(out/'TABLE_CONVERSION_AND_DATASET_PROVENANCE.md','''# Conversion and dataset provenance

| Converter | Requested training sources | Authorized disjoint sources | New complete dynamic G1 demonstrations | Compatible selected ACT checkpoint |
|---|---:|---:|---:|---:|
| Wrist Reference | 50 | 40 | 0 | 0 |
| Interaction Ours | 50 | 40 | 0 | 0 |

Original50 overlap DEV35 in7 actual recordings. The79-recording pool contains only44 non-DEV recordings; four were previously excluded and are not automatically authorized. Authorized shortfall10; even promoting all excluded recordings would leave an absolute shortfall6. No source was duplicated, promoted, newly collected, or silently removed to manufacture50/50.

Existing ACT40 datasets each contain27,577 rows from the same40 sources. Every state row exactly equals that method's previous command (first row repeats first command), and all40 videos match frozen ALOHA assets. These are not target-camera/measured-state dynamic demonstrations. Full tensor, video, checkpoint and processor hashes are in effective_data_audit/EFFECTIVE_SUPERVISION_AUDIT.json.
''')
    ablation=[dict(dev_position=i,source_id=split['evaluation_source_ids'][i],ours='NOT_ATTEMPTED_UPSTREAM',
        no_coupling='NOT_ATTEMPTED_UPSTREAM',paired_physical_observation=0) for i in split['ablation_positions']]
    csv_write(out/'TABLE_REFERENCE_COUPLING_ABLATION_PAIRED10.csv',ablation)
    atomic_text(out/'TABLE_REFERENCE_COUPLING_ABLATION_PAIRED10.md','''# Separate reference-level coupling study

Predeclared paired N=10, positions0,4,8,11,15,19,23,26,30,34. Physical paired N=0;20 reference attempts have not begun. No ACT outcome is reused as a reference outcome. No physical handoff/full-task effect is measured.

One TRAIN-source algebraic diagnostic uses one fixed contact pair and two identical seeds per setting. Enabling only cross-hand consistency yields two stationary compatible goals; disabling it yields two inconsistent predictions. This establishes a nonredundant optimization mechanism in that bounded problem, not physical coupling attribution, policy improvement, or independent two-episode evidence.
''')
    return n


def report(out):
    n=read(out/'NUMERIC_SUMMARY.json');split=read(out/'SPLIT_CONTRACT.json');sid=n['source_id'];latest=Path(n['current_final_plan']['path'])
    dependencies=[('Source identity/splits','outputs/final_single_variable_ab/00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json','All79 raw content hashes and recording identities verified','Reuse; no split invention'),
        ('Source events/registration/TCP','tools/contact_coordination/source_phase.py','Existing corrected events and registered priors; explicit unknown contacts/forces','Reuse six fixed TRAIN records'),
        ('FK and joint mapping','tools/doll_handoff_retargeting/models.py','Independent full-state FK/runtime body pose tests','Reuse named robot model'),
        ('Object/hand physics','configs/dex3_simple_graspable_doll_grasp_v2.json','Fresh bilateral controls passed; fixed20g proxy and calibrated fingers','Unchanged'),
        ('Joint bounds','outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json','Named28 order and raw readback checked','Unchanged'),
        ('Dynamic PhysX engine','tools/run_doll_handoff_graspable_proxy_v2_isaac.py','Five new dynamic traces; no object transport writes','Read-only instrumentation and explicit policy adapter'),
        ('Demonstration hand control','tools/direct_physical_execution_layer.py','Sustained enclosure before lift and giver-release order verified','Reuse unchanged for reference generation'),
        ('Planner/retiming','tools/contact_coordination/planner.py','17 regressions; goal/edge validation and authoritative bounds','Reuse'),
        ('Collision checker','tools/contact_coordination/runtime_hulls.py','Cooked contact surfaces match measured contact points within2 micrometres','Explicit cooked vertex bundle'),
        ('Target morphology','tools/contact_coordination/morphology_repair.py','Loaded waist FK verified','Reuse full-state transform adapter'),
        ('Acquisition goals','tools/contact_coordination/acquisition_plan.py','Fresh source-conditioned acquisition/lift/release passed','Explicit calibration input'),
        ('Coupled handoff goals','tools/contact_coordination/handoff_repair.py','Same on/off factors except six cross-hand residuals','Fixed qualified contact option; immutable input hash'),
        ('Segment goals','tools/contact_coordination/full_task_plan.py','Seven saved bounded versions; PLACE remains unsolved','Source carry orientation, geometry-bounded bin region'),
        ('Official ACT worker','tools/act_b_inference_worker.py','Actual A/B checkpoint inference and normalization roundtrip','Reuse existing worker for both'),
        ('Policy bridge','tools/run_act_b_isaac_diagnostic.py','Only existing ACTBridge class reused; old kinematic scene not used','Thin class adapter'),
        ('Common policy servo','tools/common_jerk_limited_otg.py','Ruckig0.19.4 outputs and named limits verified','Reuse command-state OTG; log all projections'),
        ('Camera','tools/deployment_camera_config.py','640x480 actual simulator RGB verified','Same source-like calibrated camera convention'),
        ('Measured renderer','tools/render_final_episode_registered_physical_evidence.py','Reads actual measured joints/object states','Reuse with full articulation adapter'),
        ('Independent scorer','tools/finalize_common_dex3_grasp_qualification.py','Contact, lift, retention, natural release and raw numerical checks','Unchanged')]
    atomic_text(out/'REUSE_MAP.md','# Reuse map\n\n| Component | Exact file | Compatibility evidence | Action |\n|---|---|---|---|\n'+''.join(f'| {a} | `{ROOT/b}` | {c} | {d} |\n' for a,b,c,d in dependencies))
    atomic_json(out/'REUSED_COMPONENT_HASHES.json',[record(ROOT/b) for _,b,_,_ in dependencies])
    atomic_text(out/'METHOD_PARITY.md','''# Method parity status

Main Wrist-versus-Ours dataset/policy parity is NOT ESTABLISHED: the repaired Wrist pilot, full Ours task, matched dynamic datasets, training pair and final freeze have not occurred. The existing calibrated Wrist references are retained; no generic script or Ours contact solution was substituted for A.

The inherited pure target tests cover role changes, relevant source relation changes and common SE(3) changes. Repaired morphology tests cover fixed tool application once, source scene changes, loaded named FK and the coupling residual. These are component tests, not proof that the unfinished complete A/B pipeline passes every requested parity check. The current physical task assumes source roles left giver/right receiver; other-role physical execution is not qualified.

The current TRAIN handoff fit evaluates the same qualified contact pair, two seeds, max180 evaluations per solve, unary costs, object/workspace region and geometry for both settings. Only coupling_residual is zeroed. Separate predicted left/right objects make the factor nonredundant; identical prescribed objects make it redundant. The common backend does not rerun coupled selection to rescue an inconsistent off solution. There is no physical paired10 comparison.

Both legacy ACT interface diagnostics use the same current dynamic scene, natural open-hand initial state, camera, absolute28-joint semantics, official50-step action queue (E0), no temporal ensemble, per-checkpoint processors, Ruckig servo and collision checks. Checkpoints differ, as expected; neither is compatible with the required new supervision. No source reference, event clock, converter, grasp-state machine or future image commands either policy.
''')
    atomic_text(out/'FIRST_FAILURE_CHAIN.md',f'''# First causal failures and repairs

The previous authoritative prototype remains preserved at `{ROOT}/outputs/contact_coordination_hybrid/20260907T083735Z`. Its full source task failed; actual right-side retention followed by a drop was not converted into success.

1. **Frame/calibration:** the28-joint-only FK omitted loaded waist deflection, producing approximately8–9 mm world-wrist bias. Fresh full-state standalone captures now provide measured robot-relative contact assets. Fixed TCP transforms are applied once, with T_AB mapping B into A.
2. **Offline/runtime geometry:** the authored hull overestimated the cooked object surface. Direct cooking requests on instanced child/parent prims failed; a separate in-memory replica resolved inherited geometry/APIs without editing runtime shapes. All45 mesh replicas cooked. Actual held contact points agree within2 micrometres, whereas authored hull errors reached3.8 mm at those contacts. Full measured audits now find zero forbidden samples in both standalone controls, fresh source acquisition and archived receiver-approach/ownership window. Tiny table grazing remains raw diagnostic data under the unchanged3 mm measured environment-contact rule. Same-hand legacy exclusions are an explicit limit.
3. **Contact pair:** pairing the fresh mirrored standalone contacts produced22.134/22.096 mm thumb-thumb overlap. Neither was executed. The first fit artifact key omitted calibration identity and was accidentally overwritten during TRAIN development; its exact failed fit was reconstructed and marked as such. All physical traces and all historical outputs were preserved. Input-dependent keys now refuse overwrite.
4. **Receiver contact:** reuse the archived TRAIN diagnostic's first full second after measured right ownership as a robot-relative contact/seed calibration. Full loaded FK is used; no world trajectory is copied. The later archived drop remains a failure. Two coupled goals now realize both hands consistently without unwanted stationary collisions; the off fits remain inconsistent.
5. **Transport/placement goal construction:** a calibration-transform ratio was incorrectly used as a world object orientation. Preserve the source-conditioned carried orientation, including the documented half-turn long-axis symmetry. A small pitch/height bank and orientation-specific projected XY margins create feasible bin approach goals. The object is still dynamic in physics.
6. **Current remaining failure: PLACE.** Latest bounded results: `{latest}`. Both seeds reach valid LEFT_CARRY, RECEIVER_APPROACH, GIVER_CLEARANCE and RIGHT_TRANSPORT goals, but no accepted PLACE connection exists. The bank preserves local hand/object contact and source long-axis semantics, checks full geometry, allows0/±45 degree pitch,9 orientation-specific XY positions, and two valid interior heights. Candidates fail actual named SE(3) IK tolerance or collide with the bin wall, including right middle-finger geometry. No shoulder-radius test or global impossibility claim is used.

| Phase | Current planning evidence | Newly measured source execution |
|---|---|---|
| PREGRASP / LEFT_ACQUISITION | Continuous natural approach; full hand sweep validated | Real acquisition, frame179 confirmed |
| LIFT | Continuous and retimed |60.5057 mm;1 s retention |
| LEFT_CARRY / RECEIVER_APPROACH | Valid model goals and sampled edges | NOT_ATTEMPTED in new source run |
| DUAL_SUPPORT / GIVER_RELEASE | Consistent paired goal and giver-opening sweep; archived measured receiver retention used as calibration | NOT_ATTEMPTED in new source run |
| RIGHT_CARRY / bin approach | Feasible shared backend connection | NOT_ATTEMPTED |
| PLACE | No admissible bounded complete connection | NOT_ATTEMPTED |

The fresh source run deliberately ends after lowering and natural release. It is a successful short integration check, not a full task. Full planning failures were not sent to physics. Exact world object/hand goals, fingers, q, IK errors and collision pairs/locations are saved per version under full_task_connection; no failure was threshold-clipped.
''')
    audit=read(out/'effective_data_audit/EFFECTIVE_SUPERVISION_AUDIT.json')
    lines=[]
    for method in ('A','B'):
        d=audit['methods'][method];model=next(x for x in d['checkpoint_files'] if Path(x['path']).name=='model.safetensors')
        logical=hashlib.sha256(json.dumps(dict(sources=d['sources'],action=d['canonical_action_sha256'],state=d['canonical_state_sha256'],time=d['timestamps_sha256'],videos=[x['actual']['sha256'] for x in d['videos']]),sort_keys=True).encode()).hexdigest()
        lines.append(f'ACT_{method}_CHECKPOINT_AND_DATASET_HASH: NO_COMPATIBLE_SELECTED_CHECKPOINT; legacy model {model["sha256"]}; legacy semantic data signature {logical}')
    atomic_text(out/'FINAL_REPORT.md',f'''# Hybrid retargeting + required ACT study: prerequisites unresolved

**{TERMINAL}**

The converter repair produced a verified source-conditioned acquisition/lift/release and corrected two demonstrated common defects: loaded-waist calibration and authored-versus-cooked collision geometry. The full task is still not demonstrated because placement remains invalid within the recorded bounded search. The requested disjoint50/35 split is unavailable. No matched target-image/measured-state demonstration datasets or compatible ACT training pair have been produced.

Actual legacy ACT inference has been integrated into the current dynamic simulator. A executed101 commands with model queries at0,50,100. B executed28 commands and the common collision checker stopped the next command at the bin wall. Saved RGB/state/model-output/command alignment was independently verified; maximum observation-state alignment error was2.98e-8 rad and command/raw-chunk errors were zero. These short TRAIN diagnostics do not estimate task success and do not count toward ACT-A35/B35.

```
TASK: existing doll handoff-and-bin
COMMON_CONTROL: VERIFIED (bounded component controls; scope below)
SOURCE_CONDITIONED_FULL_TASK: NOT_DEMONSTRATED
TRAINING_REQUESTED_PER_METHOD:50
TRAINING_ACTUAL_PER_METHOD:0 current compatible demonstrations;40 legacy sources per existing incompatible checkpoint
AUTHORIZED_TRAINING_SOURCES:40
TRAIN_EVAL_SOURCE_OVERLAP:0 for TRAIN40/DEV35;7 for original50/DEV35
{chr(10).join(lines)}
ACT_RUNTIME_MODE: CURRENT_RGB_MEASURED_STATE_TO_ACT_TO_COMMON_OTG; verified only with legacy incompatible checkpoints
ACT_A_PHYSICAL_TRIALS:0 /35 required primary trials; one separate101-command TRAIN diagnostic
ACT_B_PHYSICAL_TRIALS:0 /35 required primary trials; one separate28-command TRAIN diagnostic with safety stop
ACT_A_FULL_TASK_SUCCESS: NOT MEASURED
ACT_B_FULL_TASK_SUCCESS: NOT MEASURED
B_MINUS_A: NOT ESTIMABLE
REFERENCE_COUPLING_ABLATION: paired physical N=0/10; nonredundant TRAIN algebraic toggle only
NEW_ACT_TRAINING: NOT_RUN_PREREQUISITES_UNRESOLVED (required, not optional)
VLA_AND_REAL_HARDWARE: NOT_RUN_BY_DESIGN
PHYSICAL_STUDY: NOT_EXECUTED (required main comparison; partial engineering physics exists)
REPORT_PACKAGE: PARTIAL
```

There are79 actual raw recordings. TRAIN40 and DEV35 are disjoint. Original50 includes7 DEV recordings. Four other recordings were excluded previously; they are not silently promoted. Even all79 cannot supply85 disjoint sources. The smallest researcher decision is authorization for a smaller matched TRAIN40/DEV35 study, or retaining50 pending additional eligible data. No answer has been assumed. The exact IDs/content hashes are in SPLIT_AUDIT.csv and SPLIT_CONTRACT.json.

Common control evidence comprises two fresh325-frame bilateral acquisition/lift/retention/release controls, one fresh436-frame natural source acquisition, and a hash-compatible archived receiver approach, giver release and right-only retention window. New source lift was60.5057 mm with sustained contact,1 s elevated retention and natural release. No attachments, hidden carrying forces, arm rescue, post-initialization object-pose writes or clipped measured states were used. These checks establish bounded component capability, not whole-task reliability. The renderer is a rigid-object visual replay, not validated soft-body physics.

Seven full-connection versions are preserved with phase targets, candidate records and exact recoverable implementation hashes. The latest completes four connecting phases after lift and rejects PLACE. Read FIRST_FAILURE_CHAIN.md for geometry pairs, semantics and the narrow remaining issue. Exact implementation snapshots are under code_versions; all seven planning hashes match their recorded versions. An accidental overwrite of the first *new-run offline contact-fit* artifact was repaired by a labeled deterministic reconstruction; it is not represented as an original captured artifact. No physical outcome was discarded or rerun to obtain success.

The existing ACT40 pair has the same source membership,27,577 rows per method, identical source-camera video assets and the same100,000-step training protocol (saved step100000, batch8, seed1000). Actual consumed state is exactly lagged target command, not measured state. Both therefore require retraining on the authorized dynamic G1 datasets after full generation works. The separate G1-visual B-only adaptation used kinematic ownership reconstruction; the paired51 policies concern another task. Neither supplies the requested pair. EFFECTIVE_SUPERVISION_DIFF.md records canonical semantic evidence, not a generic compatibility flag.

The paired10 reference ablation has not begun. On one TRAIN source, enabling only cross-hand residuals gives two compatible stationary goals; disabling them leaves approximately47–51 mm disagreement. Candidate seeds are not independent episodes. This is neither physical coupling attribution nor an ACT learning effect. The repaired Wrist full-task integration and fixed five-source pilot are still pending, so METHOD_PARITY.md does not claim final A/B parity or a freeze.

Primary tables retain all70 intended cases as NOT_ATTEMPTED_UPSTREAM. Physical TSR is NOT MEASURED, not0%. No paired effect, confidence interval or significance result is available. DEV35 remains development evaluation. Figures explicitly show unavailable policy outcomes; reference feasibility is never relabeled as ACT success. The required35-split ACT videos and paired-reference clip cannot be generated from nonexistent trials; actual short measured replays are provided separately.

REUSED_COMPONENTS: exact files/hashes in REUSE_MAP.md and REUSED_COMPONENT_HASHES.json.
MODIFIED_COMPONENTS: acquisition calibration input; handoff contact option and input-dependent cache; cooked checker object vertices; source-conditioned transport/placement regions; thin read-only capture, causal ACT adapter, split/supervision audits and report/resume helpers. Runtime physics and qualified demonstration controller parameters remain unchanged. The five pre-existing tracked edits are preserved byte-for-byte against the initial patch.

The next engineering step is a source-preserving receiver grasp/placement connection that avoids the bin wall, followed by a real full task and the preselected TRAIN pilot. Do not generate a large training batch, select a checkpoint from DEV35 outcomes, reuse the legacy datasets as target-state data, or launch the primary policy study before those prerequisites and the split decision are resolved. REPRODUCE.md documents the working commands and fail-closed stage guards.
''')
    atomic_text(out/'METHODS_DRAFT_KO.md',f'''# 방법 초안 — 구현 범위와 미완료 단계

기존 ALOHA 인형 인계·쓰레기통 과제를 유지하였다. 실제 녹화 ID와 원본 내용 해시로 TRAIN40/DEV35를 확인하였다. 요청된 학습50개와 평가35개의 비중복 구성은 현재79개 원본으로 불가능하다. TRAIN40과 DEV35의 중복은0개이나, 최초50개 중7개가 DEV35에 포함된다. 작은 학습 집합으로의 변경은 승인되지 않았다.

좌표계는 T_AB가 B 좌표를 A 좌표로 변환하는 방식이다. 물체 자세 X와 손 안 물체 변환 G로 손 목표 X·G⁻¹를 만들고, 고정 기능 TCP–손목 변환을 한 번 적용한다. 하중에 따른 허리 관절 변위를 포함한 전체 측정 관절을 사용하여 보정하였다. 원본의 사전 리프트 구간, 초기 물체 위치·장축 방향, 인계 구간과 역할을 재사용하였다. 원본 접촉력과 동적 물체 자세는 UNKNOWN이며 성공을 의미하는 관측으로 만들지 않았다.

접촉 목표, 공통 제한 IK, 연결 경로 검사와 관절 제한 기반 시간 조정을 사용한다. 인계 시 두 손이 예측하는 물체 자세의 차이를 결합 잔차로 둔다. 동일 접촉·초기값·단항 비용·예산에서 이 잔차만 비활성화한다. 현재 진단은 TRAIN 원본1개와 초기값2개이며 물리적10쌍 제거 실험이 아니다. 물체 장축의180도 모호성과 모델 형상으로 설명되는 통 접근 영역을 사용하되 접촉 관계와 충돌 검사를 유지한다. PLACE의 유효 연결은 아직 얻지 못하였다.

물리 시뮬레이션은 기존 G1/Dex3 모델,20g 강체 대용 인형,150mm 통, 접촉 제어기 및 PhysX 설정을 유지하였다. 오프라인 충돌 형상은 실제 PhysX 조리 형상과 대응시켰고 접촉점으로 확인하였다. 이는 연체·FEM 검증이 아니다. 시뮬레이터 물체·접촉 상태를 계획 및 안전 검사에 사용하는 특권 정보 접근을 명시한다.

요구된 후속 실험은 각 변환기의 실제 G1 RGB·행동 전 측정 상태·실행 명령으로 학습한 동일 ACT 구조의 비교이다. 이 데이터와 새 학습은 아직 없다. 기존 ACT40은 ALOHA 영상과 직전 목표 명령을 상태 대용으로 소비하므로 재사용할 수 없다. 독립 인터페이스 진단에서는 현재 G1 RGB와 측정28관절만 ACT에 제공하였다. 공식50행 청크를 실행하고50프레임마다 재질의했으며, 미래 영상·원본 사건 시간표·교사 궤적·데모 접촉 상태기는 사용하지 않았다. 동일 저수준 Ruckig 및 충돌 검사가 정책 명령을 실현하였다.

본 문서는 완성된 ACT 성능 논문의 방법 절이 아니라 재현 가능한 부분 구현 초안이다. 실제 로봇, VLA 학습, 미관측 테스트 일반화 또는 양의 성능 차이를 주장하지 않는다.
''')
    atomic_text(out/'RESULTS_AND_LIMITATIONS_KO.md','''# 결과와 한계

새 원본 조건부 실행에서 인형의 실제 파지,60.5057mm 상승,1초 유지와 자연 방출을 확인하였다. 전체 과제 성공은 입증되지 않았다. 두 개의 양손 결합 목표와 통 상부까지의 연결은 생성되었으나 통 삽입 PLACE 후보에서 IK 오차 또는 손–통 벽 충돌이 남았다. 실패한 목표와 경로를 물리 엔진에 강제로 입력하지 않았다.

기존 정책을 이용한 별도 TRAIN 인터페이스 진단에서 ACT-A는101명령, ACT-B는28명령을 실행하였다. B의 다음 명령은 통 벽 충돌 검사로 중단되었다. 관측–모델 출력–실행 명령의 수치적 대응과 정규화 역변환을 검증하였다. 이 짧은 진단은 요구된35+35 과제 성공률이 아니며, A/B의 학습 효과나 우열을 추정하지 않는다.

주요 ACT-A/ACT-B 결과는 NOT MEASURED이다. 각35개 예정 사례 모두 상위 전제 미충족으로 미실행이며0% 실패로 표시하지 않는다. 원본50개 학습 계약도 충족되지 않았고 현재 호환 동적 데모 수는 각0개다. TRAIN40 승인 또는 추가 적격 데이터에 대한 연구자 결정과 전체 원본 조건부 실행이 선행되어야 한다.

결합 제거 물리 실험은 관측 쌍 N=0이다. 동일 TRAIN 원본의 두 초기값에서 보인 기하 잔차 차이는 대수적 진단일 뿐이다. DEV35는 반복 검토된 개발 평가 자료이며, 작은 초기값 수를 독립 원본 표본이나 통계적 유의성으로 해석하지 않는다. 실제 로봇·연체 역학·VLA·일반화·새 정책 학습 효과를 주장할 근거는 없다.
''')
    atomic_text(out/'METHOD_CONTRACT.md',(out/'STUDY_CONTRACT.md').read_text()+'\nCurrent status: generation and final evaluation protocols are not frozen. ACT remains required. No authorized matched50-source dataset or compatible selected pair exists.\n')
    atomic_text(out/'CHATGPT_UPDATE.md',f'''{TERMINAL}

Verified: two fresh component controls; source-conditioned acquisition/lift60.5057mm/retention/release; corrected full-state/cooked geometry; actual causal ACT interface diagnostics (A101 commands, B28 then safety stop);17 component regressions.

Unmet: source-conditioned full task (PLACE), authorized50-source split, complete matched dynamic datasets, compatible newly trained ACT pair, primary35+35 outcomes and reference10 pairs. No primary physical TSR exists. No automatic message delivery to a ChatGPT thread is provided.
''')
    return dict(status=TERMINAL,report_package='PARTIAL')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();analysis(a.run_dir);print(report(a.run_dir))
