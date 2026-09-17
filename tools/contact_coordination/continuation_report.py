"""Evidence-only continuation accounting; never launches replacement rollouts."""
import csv,io,re
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text

TERMINAL='HYBRID_RETARGETING_PROTOTYPE_REPORT_ONLY'
PHYSICAL_STAGES=['GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','TRANSPORT','BIN_ENTRY','BIN_SETTLE','FULL_TASK']


def csv_file(path,rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);atomic_text(path,stream.getvalue())


def analysis(out,resume=False):
    selection=read(out/'bootstrap/SELECTION.json');sid=selection['prototype_source_id'];root=out/'prototype'/sid
    entries=read(out/'bootstrap/SPLITS.json')['entries'];train={r['source_recording_id'] for r in entries if r['TRAIN40']};dev={r['source_recording_id'] for r in entries if r['DEV35_DIAGNOSTIC35']}
    assert set(selection['train_source_ids'])<=train and set(selection['dev_source_ids'])==dev and not train&dev
    assert selection['ablation_dev_positions']==np.round(np.linspace(0,34,10)).astype(int).tolist()
    attempts=[]
    for inv in sorted(root.glob('**/INVOCATION.json')):
        folder=inv.parent
        if not (folder/'engine.log').exists():continue
        invocation=read(inv);trace=folder/'event_log.npz'
        if not trace.exists():trace=folder/'ABORT_MEASURED_CHECKPOINT.npz'
        if not trace.exists():continue
        data=np.load(trace);frames=int(data['control_frame'][-1])+1
        score=read(folder/'HYBRID_SCORE.json') if (folder/'HYBRID_SCORE.json').exists() else None
        diagnostic=invocation.get('read_only_telemetry_diagnostic',False)
        short='morphology_acquisition_v4' in folder.parts
        terminal=(score['terminal'] if score else read(folder/'EXTERNAL_VALIDITY_ABORT.json')['terminal'] if (folder/'EXTERNAL_VALIDITY_ABORT.json').exists() else 'INFRASTRUCTURE_INVALID')
        process=read(folder/'PROCESS.json') if (folder/'PROCESS.json').exists() else {}
        attempts.append(dict(source_id=sid,attempt=str(folder.relative_to(out)),purpose='READ_ONLY_TELEMETRY_DIAGNOSTIC' if diagnostic else 'ACQUISITION_LIFT_RELEASE_DIAGNOSTIC' if short else 'TRAIN_FULL_TASK_DEVELOPMENT',
            terminal=terminal,executed_control_frames=frames,executed_physical_s=frames/30.,full_task_success=False,
            certified_geometry=False,measured_grasp=score['stages']['GRASP'] if score else True if short else 'NOT_SCORED',
            measured_lift=score['stages']['LIFT'] if score else True if short else 'NOT_SCORED',
            sensor_handoff=score['stages']['HANDOFF'] if score else 'NOT_SCORED',
            right_only_opposing_s=score['right_only_opposing_s'] if score else '',
            first_failure=score['first_failed_stage'] if score else 'FINAL_SUMMARY_WRITE' if short else 'RECEIVER_CLOSING_GEOMETRY',
            wall_s=process.get('wall_seconds',''),trace_sha256=record(trace)['sha256'],trace_path=str(trace),score_path=str(folder/'HYBRID_SCORE.json') if score else '',
            geometry_status=score.get('geometry_certification','UNRESOLVED') if score else 'UNRESOLVED_28D_TRACE_OMITS_LOADED_WAIST'))
    assert len(attempts)==5 and sum(r['purpose']=='TRAIN_FULL_TASK_DEVELOPMENT' for r in attempts)==3
    csv_file(out/'PROTOTYPE_ATTEMPTS.csv',attempts);atomic_json(out/'PROTOTYPE_ATTEMPTS.json',attempts)
    ledger=[]
    for i,s in enumerate(selection['dev_source_ids']):
        for method in ('WRIST_REFERENCE','INTERACTION_OURS'):
            ledger.append(dict(schedule_index=len(ledger),dev_position=i,source_id=s,condition=method,scheduled=1,planning_attempted=0,plan_constructed=0,physically_run=0,success=0,unknown_infrastructure=0,terminal='NOT_ATTEMPTED_UPSTREAM',physical_stages='NOT_ATTEMPTED',physical_TSR='NOT_MEASURED'))
    for i in selection['ablation_dev_positions']:
        ledger.append(dict(schedule_index=len(ledger),dev_position=i,source_id=selection['dev_source_ids'][i],condition='OURS_NO_COUPLING',scheduled=1,planning_attempted=0,plan_constructed=0,physically_run=0,success=0,unknown_infrastructure=0,terminal='NOT_ATTEMPTED_UPSTREAM',physical_stages='NOT_ATTEMPTED',physical_TSR='NOT_MEASURED'))
    csv_file(out/'PER_INSTANCE_RESULTS.csv',ledger)
    main=[dict(condition=m,scheduled=35,planning_attempted=0,plan_constructed=0,physically_run=0,certified_valid_rollouts=0,success=0,unknown_infrastructure=0,not_attempted_upstream=35,plan_rate='NOT_MEASURED',cumulative_observed_success='NOT_MEASURED',conditional_physical_TSR='NOT_MEASURED',confidence_interval='NOT_APPLICABLE_N0') for m in ('WRIST_REFERENCE','INTERACTION_OURS')]
    csv_file(out/'TABLE_MAIN_WRIST_VS_OURS.csv',main)
    atomic_text(out/'TABLE_MAIN_WRIST_VS_OURS.md','''# Wrist versus Ours — DEV35 accounting, comparison not executed

| Condition | Scheduled | Planning attempted | Plans | Physical runs | Successes | Unknown/infrastructure | Upstream not attempted | Physical TSR |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| WRIST_REFERENCE | 35 | 0 | 0 | 0 | 0 | 0 | 35 | NOT MEASURED |
| INTERACTION_OURS | 35 | 0 | 0 | 0 | 0 | 0 | 35 | NOT MEASURED |

The protocol was not frozen. These are intended-instance accounting counts, not 35 failed trials per method. Plan rate, observed pipeline completion, conditional physical completion, confidence intervals and paired effects are not estimable. No-plan outcomes were not fabricated. With no observations, every completion count from 0 to 35 remains possible; this is a logical bound, not a confidence interval. DEV35 is development evaluation, not untouched testing.
''')
    ablation=[dict(dev_position=i,source_id=selection['dev_source_ids'][i],ours='NOT_ATTEMPTED_UPSTREAM',no_coupling='NOT_ATTEMPTED_UPSTREAM',paired_physical_observation=0) for i in selection['ablation_dev_positions']]
    csv_file(out/'TABLE_COUPLING_ABLATION_PAIRED10.csv',ablation)
    atomic_text(out/'TABLE_COUPLING_ABLATION_PAIRED10.md','''# Coupling — predeclared paired-10 accounting

Intended paired N=10; observed paired physical N=0. Rounded linspace positions: 0, 4, 8, 11, 15, 19, 23, 26, 30, 34. The CSV gives exact recording identities. Ours results would be reused from the main study, with no success-selected rerun. No physical effect or confidence interval is available.

The TRAIN algebraic diagnostic removes only cross-hand predicted-object pose residuals. It is nonredundant for independently optimized object poses and becomes redundant for identical fixed shared poses. The six optimizer candidates per setting are not independent source episodes or a physical paired-10 experiment.
''')
    latest=root/'full_task_connection/ead64a3c8365/physics_attempt_03';telemetry=root/'full_task_connection/ead64a3c8365/geometry_telemetry_diagnostic'
    audit=read(telemetry/'POSE_PARITY_AUDIT.json');short=read(root/'morphology_acquisition_v4/physics_attempt_01/INDEPENDENT_TRACE_AUDIT.json')
    numerical=dict(terminal=TERMINAL,source_id=sid,development_physical_executions=5,full_task_development_attempts=3,full_task_successes=0,
        source_conditioned_full_task='NOT_DEMONSTRATED',short_measured_lift_m=short['lift_m'],latest_full_attempt_score=read(latest/'HYBRID_SCORE.json'),
        max_full_named_fk_position_error_m=audit['maximum_full_named_fk_position_error_m'],max_full_named_fk_rotation_error_rad=audit['maximum_full_named_fk_rotation_error_rad'],
        main=main,ablation_observed_paired_N=0,protocol_frozen=False,DEV35_started=False,ACT_VLA_training='NOT_RUN_BY_DESIGN',attempts=attempts)
    atomic_json(out/'NUMERIC_SUMMARY.json',numerical)
    from .mechanism_audit import run as mechanism
    mechanism(out)
    atomic_json(out/'prototype/RESULT.json',dict(status='SOURCE_CONDITIONED_FULL_TASK_NOT_DEMONSTRATED',measured_partial_task=True,summary=record(out/'NUMERIC_SUMMARY.json'),prerequisite='Complete-articulation contact calibration and valid handoff withdrawal/transport remain unresolved'))
    return dict(status='PROTOTYPE_NUMERIC_ACCOUNTING_VERIFIED',intended_DEV_instances=80,attempted_DEV_instances=0,development_physical_executions=5)


def report(out,resume=False):
    n=read(out/'NUMERIC_SUMMARY.json');sid=n['source_id'];score=n['latest_full_attempt_score'];cal=read(out/'target_repair/CALIBRATION_VALIDITY.json')
    mechanism=read(out/'MECHANISTIC_DIAGNOSTICS.json')
    reused=[]
    for value in re.findall(r'/home/jbnu/[^\s|`]+',(out/'REUSE_MAP.md').read_text()):
        path=Path(value.rstrip('.,;'))
        if path.is_file() and str(path) not in [v['path'] for v in reused]:reused.append(record(path))
    atomic_json(out/'bootstrap/REUSE_COMPONENTS.json',reused)
    atomic_text(out/'FINAL_REPORT.md',f'''# Hybrid continuation — measured prototype report

**{TERMINAL}**

The converter now produces natural-start manipulation commands and actual acquisition, lift, carry and temporary receiver support. No source-conditioned full task was demonstrated. Three full TRAIN attempts and two diagnostics were preserved. The planned DEV35 comparison and paired-10 ablation were not started.

TASK: existing doll handoff-and-bin  
COMMON_CONTROL: UNVERIFIED for complete current geometry certification; previous standalone acquisition/release numerical checks match dependencies, but the scripted handoff has authored thumb-hull overlap and incomplete articulation telemetry. Historical PASS is not reused as full qualification.  
SOURCE_CONDITIONED_FULL_TASK: NOT_DEMONSTRATED  
WRIST_DEV35: scheduled 35 / planned 0 / physically run 0 / success 0 / unknown-infrastructure 0 / upstream not attempted 35  
OURS_DEV35: scheduled 35 / planned 0 / physically run 0 / success 0 / unknown-infrastructure 0 / upstream not attempted 35  
COUPLING_ABLATION: intended paired N=10 / observed paired N=0 / algebraically meaningful toggle / physical outcome NOT MEASURED  
ACT_VLA_TRAINING: NOT_RUN_BY_DESIGN  
PHYSICAL_STUDY: NOT_EXECUTED (DEV comparison); prototype physics was executed  
REPORT_PACKAGE: PARTIAL (prototype evidence complete; physical comparison/replays unavailable)

The source unit is `{sid}`, the original deterministic TRAIN prototype. Five additional TRAIN IDs were fixed before new outcomes; all six source phase records were recovered, including one single-sample pre-lift estimate explicitly marked uncertain. The five-episode physical pilot was not reached. TRAIN40 is a calibration pool; DEV35 remains development evaluation.

## Actual development evidence

| Execution | Observed result | Limit |
|---|---|---|
| Natural acquisition/lift/release diagnostic | {1000*n['short_measured_lift_m']:.2f} mm measured lift; 1 s opposing table-free retention; 1.5 s natural release/table support | Final summary writer failed after complete trace save; raw trace retained; no synthetic replacement summary |
| Full attempt 01, translated contact, closing at arrival | Real acquisition/lift/carry; receiver closing caused forbidden thumb overlap | Explicitly aborted; unchanged intermediate finger sweep had been omitted from planning validation |
| Full attempt 02, closing advanced 90 frames | Real acquisition/lift; receiver contact lost after giver release | Bounded ownership guard stopped transport; raw angular speed 61.17 rad/s exceeds the fixed 50 rad/s artifact bound |
| Full attempt 03, rotated receiver contact | Measured sensor handoff and {score['right_only_opposing_s']:.3f} s right-only opposing support; drop outside bin during giver withdrawal | No transport/bin completion. Existing independent numerical scorer calls task failure; complete geometry certification remains unresolved |
| Read-only telemetry diagnostic | Exact first 1,020 commands of attempt 03, fresh natural start; same loss reproduced | Additional diagnostic, never substituted for the primary attempt or counted as a new successful source instance |

`PROTOTYPE_ATTEMPTS.csv` lists all five executions and trace hashes. `PER_INSTANCE_RESULTS.csv` retains all 80 intended DEV method-instances as NOT_ATTEMPTED_UPSTREAM. No later physical stages are fabricated. Zero physical DEV observations means TSR NOT MEASURED, not 0%. The observed TRAIN versions are adaptive engineering trials on one source, not independent Bernoulli samples; a success-rate confidence interval or paired significance test would be misleading.

## What was repaired

The prior converter could fit an acquisition wrist point while putting the modeled palm about 20.72 mm into the object. The repair separates source gripper meaning from Dex3 contact morphology using T_WH = X inverse(T_HO), with the calibrated fixed wrist/tool transform applied exactly once. Source initial pose/yaw, events and handoff priors determine the task scene and goals. Calibration supplies contact relations and deterministic seeds, not the successful scripted world trajectory.

The shared existing SE(3) backend now checks candidate connections before selection. Full arm edges, nominal closing sweeps, giver opening, carried geometry and post-retiming commands were checked. The 150 mm bin and its existing 3 mm bevel are reconstructed with the exact runtime function. The same hand primitive closes earlier during receiver approach; a common phase clock prevents early giver release. There is no spatial arm rescue, attachment, task restart within a run, or object pose write after initialization.

The final diagnostic localized another common defect: 28-joint FK omitted loaded waist motion. At frame 900, waist pitch was about 0.01850 rad; reconstructed left/right wrists differed from runtime poses by 8.22/9.31 mm. Full named measured FK reduces the maximum error across checked bodies/frames to **{1e6*n['max_full_named_fk_position_error_m']:.3f} micrometres** and {n['max_full_named_fk_rotation_error_rad']:.3g} rad. Full articulation/body telemetry is now logged; calibration extraction rejects incomplete traces. This fixes reconstruction, not manipulation reliability. The historical contact calibration must be recaptured with the complete state and geometric qualification before another freeze.

## Smallest unresolved issue and bounded stop

Robot-relative contact calibration was derived from incomplete articulation states. The receiver candidate can sustain stationary opposing contact but loses the doll during giver withdrawal. The diagnostic also retains a discrepancy between authored convex-hull overlap and PhysX contact separations; cooking/contact margins and the checker's broad same-hand exclusions prevent a complete solid-geometry certificate. No global impossibility or exhaustive reachability claim follows.

The full-state diagnostic flags three authored-hull palm/object overlap samples at frame 785, maximum 3.046 mm, while the runtime contact separation differs. Its geometry-augmented scorer therefore reports EXECUTION_ABORT_PHYSICAL_VALIDITY; the original attempt's numerical task-failure result remains preserved. No sensor-derived handoff duration is promoted to a geometrically qualified handoff.

The selected source-handoff goal was relocated {1000*mechanism['handoff_relocation_m']:.1f} mm from the raw source-inferred prior. Full planned duration was {mechanism['planned_duration_s']:.2f} s, or {mechanism['time_stretch']:.3f} times the source duration. These are recorded adaptations under the declared task region, not high-fidelity tracking claims. `MECHANISTIC_DIAGNOSTICS.json` preserves both raw and selected goals, handoff wrist deviations, actual full-state contact-relation errors, predicted cross-hand consistency and all phase times. The incomplete-state calibration limits interpretation of those relation errors.

The productive development bound comprised morphology correction, calibrated contact-region fitting, closing-sweep timing repair, a receiver contact rotation strategy, and one unchanged-command telemetry diagnostic. Further blind seeds, physics tolerance changes or 80 upstream failure cards would not resolve the identified calibration requirement. M2 is not met; the study freeze is withheld. The corrected logging/FK implementation, regression checks, exact traces and resumable report stages are delivered.

## Reuse and modifications

REUSED_COMPONENTS: exact paths and SHA-256 values are in `REUSE_MAP.md` and `bootstrap/REUSE_COMPONENTS.json`: authoritative source manifests/replacements, raw references/timestamps, corrected source events, ALOHA/G1 FK and tool registration, G1/Dex3 XML/USD, named hard limits, P14 OPEN/PRESHAPE/HOLD controls, proxy physics, 150 mm bin, SciPy IK backend, existing retiming bounds, PhysX runner/contact logger, independent numerical scorer and measured renderer.

MODIFIED_COMPONENTS: `source_phase.py` (honest single-sample evidence), `planner.py` (common candidate/edge validation and bounded goal-region search), `morphology_repair.py`, `acquisition_plan.py`, `handoff_repair.py`, `full_task_plan.py`, `runtime_hulls.py`, `physical_attempt.py`, `phase_physics.py`, `phase_clock_runtime.py`, `score_hybrid.py`, `pose_parity_audit.py`; thin continuation/report/replay entry points and small regression tests. All are under `tools/contact_coordination/`. The five originally dirty legacy tracked files remain unchanged by this continuation. Exact final code inventory is in `FINAL_CODE_MANIFEST.json`; per-attempt versions remain in saved snapshots.

## Supported and unsupported claims

Supported: source-derived goals led to actual natural acquisition/lift/carry in dynamic PhysX; measured partial handoff behavior exists; a specific loaded-articulation reconstruction error was localized and fixed. Cross-hand residuals are algebraically nonredundant on the tested candidate problem.

Unsupported: full task success, Wrist-versus-Ours performance improvement, physical coupling attribution, reliability over multiple sources, untouched test performance, calibrated soft-body physics, real-robot transfer, visual target-domain autonomy, ACT or VLA learning. The system uses privileged simulator object/contact state. The rigid plush surrogate is not FEM or validated deformable physics.

The ordinary measured replays are illustrations of named attempts, not estimates of typical behavior. No all-card videos or fabricated successful motion replace the missing DEV35 rollouts. See `REPRODUCE.md`, `PAPER_FIGURE_DIRECTION.md`, and `replays/REPLAY_STATUS.json`.
''')
    atomic_text(out/'METHOD_CONTRACT.md','''# Implemented prototype contract — not an evaluation freeze

T_AB maps B coordinates into A; distances are metres; quaternions are XYZW. Contact G_i=T_HiO implies T_WHi=X inverse(G_i), followed once by inverse(T_wristHi). Full named articulation state is required for measured FK/contact calibration. The existing common config fixes world/task registration, joint identities and units.

Source initial object pose comes from pre-motion image registration and inferred height, not wrist XYZ or a method prediction. Pre-lift tool/object relation is an inference. Object forces/contact ownership and dynamic source object orientation are UNKNOWN; no force labels are fabricated. The source handoff prior is a soft spatial cost, with task order preserved under retiming. The current morphology pipeline implements the existing left-to-right task; general role inversion is not qualified.

Ours phase contacts use independent robot morphology calibration, bounded contact candidates and a shared-pose fit. The latest TRAIN region fit used 3 contacts × 2 seeds × 180 maximum least-squares evaluations per contact/seed, identically on/off. Source position scale 1, orientation prior 0.1, upright-axis scale 10, region scale 100; coupling position/rotation scales 100/10. Named joint bounds are unchanged. Geometry screening and subsequent path planning are additional recorded work; there is no frozen equal total study budget yet.

Connecting IK uses common previous/calibrated/midpoint seeds, 120 evaluations per seed per goal, position/rotation tolerances 3 mm/0.05 rad and intermediate joint samples at no more than 0.02 rad. These are bounded local searches, not impossibility proofs. Carry uses predicted rigid contact geometry; runtime object remains dynamic. Nominal wrist fidelity is separate from hard-limit/collision admission. Empirical acceleration/manipulation limits and existing joint velocity bounds retime stop-to-stop quintics. Source durations are not mandatory.

The original shared hand controller uses 0.015 N digit contact, 0.02 N table threshold, calibrated 45-frame preshape/close/release and bounded preload. Acquisition candidate precedes lift; receiver candidate precedes giver release; sole receiver support is checked after release. A 90-frame receiver closing advance and arrival eligibility guard are common phase integration, with no online spatial arm correction. New full-state telemetry is read-only.

Measured stage contract: GRASP requires confirmed enclosing/opposing contact before commanded lift and at least 0.1 s sustained contact. LIFT requires measured 50 mm height gain, opposing contact and table-free retention for 1 s. HANDOFF requires dual opposing support for 0.1 s followed by right-only retention for 1 s. RIGHT_OWNERSHIP excludes measured left digit/palm support. TRANSPORT requires at least 50 mm measured progress toward the bin under right-only support. BIN_ENTRY uses the authoritative interior; BIN_SETTLE requires at least 1 s containment, bin support and ≤0.02 m/s. FULL_TASK additionally requires physical validity and the declared clean/premature-in-bin release classification. No acquisition is not classified as premature drop. Legacy scripted stage-name equality and a second right upward lift are not natural-task requirements.

Raw states are retained. Existing 3 mm penetration, 30 mm object-step and 50 rad/s angular-artifact limits are not widened. Post-loss linear speed is diagnostic under the reused outcome-centric scorer. A numerical validity pass is not a complete geometry certificate; waist completeness, runtime convex cooking and collision filtering limitations are separately recorded. No model/controller/scorer study freeze has occurred.
''')
    atomic_text(out/'METHOD_PARITY.md','''# Method parity — qualification remains incomplete

| Resource | Contract | Verified here |
|---|---|---|
| Sources/scenes | Same source ID, registration, object/bin and tool calibration | Six TRAIN phase records; DEV order and paired subset fixed |
| Ours coupling toggle | Only predicted-object cross-position/rotation factors disabled | Algebraic tests and same 3-contact/2-seed TRAIN banks; no physical paired run |
| Wrist spatial objective | Calibrated source wrist/TCP prior; no paired-contact optimizer | Existing reference recovered; repaired full physical integration not completed |
| Planner/executor | Method-blind goals, bounds, validator, timing/controller inputs | Interface/source tests; no hidden method-ID arm rescue |
| Source roles | Preserve source giver/receiver intent | Legacy wrapper role perturbation passes; current repaired morphology pipeline supports only the existing left-to-right task and is not role-generalization qualified |
| World coordinates | One common transform, no independent hand rebasing | Tool/contact SE(3) invariance and cross-factor norm tests pass |
| Budget/cache | Same total budget, cache accounting and search facilities | Not frozen; main-study parity is not claimed |
| Physics/geometry | Same dynamic proxy, controls, authored assets and 150 mm bin | Dependency hashes match; full geometry certification unresolved |

The shared-X consistency term is not redundant when each arm implies its own X through independently varying q. It vanishes for identical prescribed X; the unit test explicitly checks this case. Coupling-off is not an old ACT-B pipeline. Safety validation may reject inconsistent/unsafe output but must not rerun coupled contact selection. Stationary TRAIN fit diagnostics do not attribute a physical result to coupling.

Seventeen tests cover the legacy wrapper, repaired transform and phase-clock behavior, current coupling factor, and full named FK against independently logged runtime body poses. Passing them does not establish a competent executed Wrist baseline or a frozen study.
''')
    atomic_text(out/'METHODS_DRAFT_KO.md','''# 방법 초안 — 현재는 프로토타입 보고 범위

기존 ALOHA 인형 전달·휴지통 배치 과제의 녹화, FK, 기능적 TCP, 초기 장면 정합 및 수정된 이벤트를 재사용하였다. TRAIN40은 보정 자료이며 ACT 학습을 수행하지 않았다. 초기 물체 위치·방향은 동작 전 영상 정합으로 추정하였고, 손목 좌표를 물체 중심으로 대체하지 않았다. 접촉력 및 동적 물체 자세의 원본 관측은 확보되지 않았으므로 UNKNOWN으로 구분하였다.

조밀한 자유공간 손목 경로를 단계별 접촉 목표와 공간 사전값으로 변환하였다. 좌표 변환 T_AB는 B에서 A로의 변환이며, 물체의 목표 자세 X와 물체-손 관계 G로 손 목표 X G⁻¹를 구성한다. 고정 손목-도구 변환은 한 번만 적용한다. Dex3 형태 보정은 접촉 관계와 초기값을 제공하며, 성공한 스크립트의 전체 월드 궤적을 복사하지 않는다.

전달 목표는 양손이 예측하는 물체 자세의 일관성을 최적화한다. 결합 제거 조건은 같은 단항 비용, 후보, 초기값, 영역과 평가 한도를 유지하고 교차 위치·회전 잔차만 제거한다. 이번 결과는 TRAIN 한 녹화의 기구학 진단이며 물리적 결합 효과 실험이 아니다. Wrist 기준선은 기존 보정된 손목 경로를 공간 목적으로 사용하도록 정의하였으나 수리된 전체 실행 통합 검증은 완료되지 않았다.

기존 IK와 충돌 검사 기반 후보 연결 및 공통 속도·경험적 가속도 제한 재타이밍을 사용하였다. 실제 물체는 동적 강체이며 부착, 운반 힘, 초기화 이후 자세 쓰기를 사용하지 않았다. 공통 접촉 기반 손 제어기는 획득 후보 후 들기, 수신 후보 후 제공 손 열기 순서를 따른다. 시뮬레이터 접촉 및 물체 상태 접근은 특권 정보이며 시각 기반 실제 로봇 자율성을 주장하지 않는다.

완전한 측정 FK에는 팔·손 28관절 외에 하중에 의해 움직이는 허리 관절이 필요함을 확인하였다. 전체 관절과 강체 자세 로깅 및 재구성은 수정했으나 이전 보정 자료에 누락된 상태를 소급 생성하지 않았다. 따라서 전체 물리 비교를 위한 보정·기하 검증 및 프로토콜 동결은 미완료이다.
''')
    atomic_text(out/'RESULTS_AND_LIMITATIONS_KO.md',f'''# 결과와 한계 — 물리 프로토타입

전체 과제 성공은 입증되지 않았다. 고정 TRAIN 녹화 `{sid}`에서 짧은 획득·들기·놓기 실행 1회, 전체 과제 개발 시도 3회, 동일 명령의 읽기 전용 기하 진단 1회를 보존하였다. 마지막 전체 시도는 접촉 센서상 {score['right_only_opposing_s']:.3f}초 동안 오른손 단독 지지를 보였으나 제공 손 후퇴 중 물체를 떨어뜨렸다. 휴지통 진입·정착은 관측되지 않았다. 이 수치를 여러 에피소드 성공률이나 신뢰구간으로 변환하지 않는다.

첫 전체 시도의 엄지 충돌, 두 번째 시도의 수신 접촉 상실 및 각속도 초과, 세 번째 시도의 낙하를 모두 기록하였다. 종료 결과 파일 쓰기 오류가 있었던 짧은 실행도 제외하지 않았다. 측정 상태를 자르거나 실패 후 임계값을 완화하지 않았다.

허리 관절 누락은 전달 자세에서 약 8–9 mm의 손목 FK 오차를 만들었다. 전체 측정 관절을 사용한 재구성은 검사 지점에서 최대 {1e6*n['max_full_named_fk_position_error_m']:.3f} μm 위치 오차로 실제 강체 자세와 일치하였다. 그러나 기존 접촉 보정의 재취득과 PhysX 볼록체 조리·접촉 분리값의 차이 확인이 남아 있다. 수치 점수의 유효성 통과를 완전한 고체 비침투 인증으로 해석하지 않는다.

DEV35의 Wrist 35회, Ours 35회와 결합 제거 10회는 모두 상류 전제 미충족으로 미시도이다. 물리 TSR은 0%가 아니라 NOT MEASURED이며 A/B 개선, 유의성 또는 결합의 인과 효과를 주장할 수 없다. DEV35는 개발 평가 자료다. 추가 과제, 실물 로봇, 변형체 FEM, ACT 및 VLA 학습은 수행하지 않았다.
''')
    atomic_text(out/'CURRENT_STATUS.md',f'{TERMINAL}\n\nSource-conditioned full task: NOT_DEMONSTRATED. TRAIN physics: 3 full development attempts + 1 short acquisition diagnostic + 1 read-only telemetry diagnostic, all preserved. DEV35: 0/80 attempted. Complete-articulation FK/logging repaired and independently tested; full contact calibration/geometry qualification remains unresolved. Report and measured replay verification are recorded separately.\n')
    atomic_text(out/'CHATGPT_UPDATE.md',f'''{TERMINAL}

Actual source-conditioned acquisition, lift and partial handoff were measured. No full task completed. The final diagnostic identified loaded waist deflection omitted from the earlier 28D FK/calibration; complete-state logging and FK now match independent runtime body poses within 0.001 mm on checked frames. Do not resume from old incomplete calibration as if qualified. All three full TRAIN failures and both diagnostics remain saved. DEV35/paired-10 were not started, ACT/VLA was not run. See FINAL_REPORT.md and REPRODUCE.md. This file does not send messages to a ChatGPT thread.
''')
    return dict(status='PROTOTYPE_REPORT_WRITTEN',terminal=TERMINAL,physical_study='NOT_EXECUTED',report_package='PARTIAL')


def reproduce(out):
    atomic_text(out/'REPRODUCE.md',f'''# Reproduction and safe resume

Run from `/home/jbnu/aloha_g1_dataset` with `/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python`. Rendering uses `MUJOCO_GL=egl`; physical execution used the existing `/home/jbnu/miniconda3/envs/isaaclab6/bin/python` and installed IsaacLab 3 beta/PhysX. No hardware commands are involved.

The actual thin entry point recognizes the continuation through `CONTINUATION_STATE.json`. Its stage graph is inspect → failure_localization → common_control → target_repair → prototype → pilot → freeze → main_dev35 → paired_ablation → analysis → paper_report → measured_replays → final_verification. Historical aliases bootstrap/primary_dev35/report/render map to inspect/main_dev35/paper_report/measured_replays.

Verified report resume (no physical replay or new method attempts):

```bash
MUJOCO_GL=egl /home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python tools/contact_coordination/run_hybrid_study.py --run-dir {out} --stage analysis --until final_verification --resume
```

Regression tests:

```bash
/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python -m unittest tools.contact_coordination.test_hybrid tools.contact_coordination.test_morphology_repair -v
```

Independent loaded-FK audit, using the actual saved diagnostic trace:

```bash
/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python -m tools.contact_coordination.pose_parity_audit --run-dir {out} --attempt-dir {out}/prototype/GoPark_20260820_152058/full_task_connection/ead64a3c8365/geometry_telemetry_diagnostic
```

`--resume` checks source/config/code signatures and output hashes before reusing a stage. A changed report/analysis dependency regenerates derived artifacts; it never reruns a completed physical attempt. Each new physical attempt directory is immutable and the physical wrapper rejects an existing attempt name. The runner uses a nonblocking run lock and atomic stage/result writes. It does not create a background service, evade session limits, or message a ChatGPT thread. Infrastructure exceptions are persisted and raised; this entry point does not automatically retry them (therefore no infinite recovery loop).

In this bounded continuation, inspect/failure_localization/common_control/target_repair/prototype consume the saved evidence. `target_repair` reports CALIBRATION_RECAPTURE_REQUIRED. `prototype` audits preserved attempts; it does not reinterpret a failed attempt as success. Pilot/freeze/main_dev35/paired_ablation return NOT_ATTEMPTED_UPSTREAM because M2 and common calibration qualification are unmet. Downstream physical study execution is not implemented/qualified and cannot be unlocked merely by changing a status string. The entry point is a truthful resumable evidence handoff, not a claim that the study is ready.

Exact physical commands, timeouts, dependency records and raw outputs are in each `INVOCATION.json`, `DEPENDENCIES.json`, `PROCESS.json`, `engine.log` and trace. Reproduction of historical behavior requires that attempt's saved code snapshot and input hashes. Do not run those old commands against current code expecting hash-equivalent behavior. In particular, new calibration extraction and primary-execution guards reject the known-incomplete legacy contact calibration. No new physics run is required to reproduce the report or change its camera.

Resume engineering at the smallest identified requirement: recapture a geometrically qualified common contact/handoff calibration with full named measured joint state and runtime rigid-body poses; use the repaired extraction, verify actual loaded grasp/withdrawal, and regenerate source goals under a new version. The old script's cross-hand overlap is not admissible calibration. Do not synthesize missing waist trajectories or reuse its entire world task path as Ours. Then establish M2 and the fixed five-source pilot before freezing budgets and any DEV outcomes.

Source provenance: bootstrap/SPLITS.json retains authoritative raw recording identities, replacements and reference hashes; source_phase/<source_id>/ contains the six current phase records and source priors. Pre-lift uncertainty metadata was added after early attempts; the exact earlier phase-record files are preserved in bootstrap/source_phase_before_single_sample_fix. Runtime-bin150 files supersede the initial 190 mm offline export, which is retained as a diagnostic. One early three-pose planning diagnostic was overwritten during expansion to nine poses; its explicit PRIOR_3POSE_DIAGNOSTIC_NOTE.json records that limitation. No physical trace or outcome was deleted.

The previous run `{ROOT}/outputs/contact_coordination_hybrid/20260907T073437Z` remains intact. Its common-control movie is calibration illustration only and is not treated as a current geometrically qualified source-conditioned success. `FINAL_CODE_MANIFEST.json`, `FINAL_VERIFICATION.json` and each replay's JSON bind this report's actual artifacts.
''')


def verify(out):
    import hashlib,subprocess
    required=['FINAL_REPORT.md','REPRODUCE.md','METHOD_CONTRACT.md','METHOD_PARITY.md','METHODS_DRAFT_KO.md','RESULTS_AND_LIMITATIONS_KO.md','PAPER_FIGURE_DIRECTION.md','REUSE_MAP.md','FIRST_FAILURE_CHAIN.md','TABLE_MAIN_WRIST_VS_OURS.csv','TABLE_MAIN_WRIST_VS_OURS.md','TABLE_COUPLING_ABLATION_PAIRED10.csv','TABLE_COUPLING_ABLATION_PAIRED10.md','PER_INSTANCE_RESULTS.csv','PROTOTYPE_ATTEMPTS.csv']
    assert all((out/p).is_file() and (out/p).stat().st_size>0 for p in required)
    rows=list(csv.DictReader((out/'PER_INSTANCE_RESULTS.csv').open()));assert len(rows)==80 and all(r['terminal']=='NOT_ATTEMPTED_UPSTREAM' and r['physically_run']=='0' for r in rows)
    assert sum(r['condition']=='WRIST_REFERENCE' for r in rows)==35 and sum(r['condition']=='INTERACTION_OURS' for r in rows)==35 and sum(r['condition']=='OURS_NO_COUPLING' for r in rows)==10
    attempts=read(out/'PROTOTYPE_ATTEMPTS.json');assert len(attempts)==5
    for a in attempts:assert record(a['trace_path'])['sha256']==a['trace_sha256']
    media=read(out/'replays/REPLAY_STATUS.json');assert len(media['products'])==3 and not media['all_card_videos_created']
    for p in media['products']:
        assert record(p['video']['path'])==p['video'];assert record(p['dependencies']['trace']['path'])==p['dependencies']['trace']
    for name in ['Fig1_Method','Fig2_AB_Physical_Results','Fig3_Coupling_Ablation','PROTOTYPE_MEASURED_STAGES']:
        for suffix in ['png','svg']:assert (out/'figures'/f'{name}.{suffix}').stat().st_size>0
    assert 'Ran 17 tests' in (out/'TEST_RESULTS.txt').read_text() and (out/'TEST_RESULTS.txt').read_text().rstrip().endswith('OK')
    initial=(out/'bootstrap/tracked_changes_initial.patch').read_bytes();current=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
    assert current==initial,'Pre-existing tracked changes no longer match the initial checkpoint'
    source_paths=list((ROOT/'tools/contact_coordination').glob('*.py'))+list((ROOT/'configs/contact_coordination').glob('*.json'))
    atomic_json(out/'FINAL_CODE_MANIFEST.json',[record(p) for p in sorted(source_paths)])
    result=dict(status='PROTOTYPE_PACKAGE_VERIFIED',terminal=TERMINAL,DEV35_instances=80,DEV35_attempted=0,development_physical_executions=5,full_task_development_attempts=3,source_conditioned_full_task=False,
        replay_videos=3,tests_passed=17,preexisting_tracked_diff_unchanged=True,preexisting_diff_sha256=hashlib.sha256(current).hexdigest(),report_package='PARTIAL',physical_study='NOT_EXECUTED',calibration_requires_recapture=True,
        reports=[record(out/p) for p in required],media_manifest=record(out/'replays/REPLAY_STATUS.json'),code_manifest=record(out/'FINAL_CODE_MANIFEST.json'))
    atomic_json(out/'FINAL_VERIFICATION.json',result);return result


def perform_stage(out,stage,resume):
    aliases={'bootstrap':'inspect','primary_dev35':'main_dev35','report':'paper_report','render':'measured_replays'};stage=aliases.get(stage,stage)
    if stage=='inspect':
        assert (out/'bootstrap/PREVIOUS_FINAL_REPORT.md').exists();return dict(status='SAVED_INSPECTION_VERIFIED',selection=record(out/'bootstrap/SELECTION.json'))
    if stage=='failure_localization':
        assert (out/'FIRST_FAILURE_CHAIN.md').exists();return dict(status='FIRST_FAILURE_LOCALIZED',evidence=record(out/'FIRST_FAILURE_CHAIN.md'))
    if stage=='common_control':return dict(status='COMMON_CONTROL_UNVERIFIED',dependency_check=record(out/'bootstrap/CONTROL_DEPENDENCY_VERIFICATION.json'),reason='Scripted handoff geometry and complete-state contact calibration are not qualified')
    if stage=='target_repair':return dict(status='CALIBRATION_RECAPTURE_REQUIRED',evidence=record(out/'target_repair/CALIBRATION_VALIDITY.json'))
    if stage=='prototype':
        analysis(out,resume);return dict(status='SOURCE_CONDITIONED_FULL_TASK_NOT_DEMONSTRATED',physical_attempts=record(out/'PROTOTYPE_ATTEMPTS.json'))
    if stage in ('pilot','freeze','main_dev35','paired_ablation'):
        return dict(status='NOT_ATTEMPTED_UPSTREAM',reason='Source-conditioned full task and common calibration qualification not established; downstream physical study runner is not qualified')
    if stage=='analysis':return analysis(out,resume)
    if stage=='paper_report':
        result=report(out,resume);reproduce(out)
        from .continuation_media import figures,direction
        figures(out);direction(out);return result
    if stage=='measured_replays':
        from .continuation_media import replays
        return replays(out,resume)
    if stage=='final_verification':return verify(out)
    raise ValueError('Unsupported continuation stage: '+stage)
