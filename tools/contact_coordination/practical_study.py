"""Immutable TRAIN40 identity, recipes and decision rules for one practical run."""
from pathlib import Path
from datetime import datetime,timezone
import shutil
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text,fingerprint
from .practical_parameters import RECIPES,DEFAULT,BOUNDS

BASE=ROOT/'outputs/source_guided_rrt_rebuild/20260911T035757Z'
METHODS=('B_INDEPENDENT','C_COUPLED')
FIXED_BACKEND=('source_phase.py','source_contract.py','planning_kinematics.py','planner.py',
    'joint_path_planner.py','path_quality.py','source_motion_prior.py','handoff_repair.py',
    'full_task_plan.py','shared_handoff_candidates.py','loaded_contact_geometry.py',
    'phase_clock_runtime.py','full_attempt_runtime.py','full_attempt.py','full_attempt_physics.py',
    'execution_timing.py','contact_command_retiming.py','score_hybrid.py','abc_score.py')


def now():return datetime.now(timezone.utc).isoformat()


def stage_inputs(base,out):
    for name in ('bootstrap','target_repair','source_phase'):
        shutil.copytree(base/name,out/name,dirs_exist_ok=True)
    for name in ('SPLIT_CONTRACT.json','COVERAGE8.json'):
        shutil.copy2(base/name,out/name)


def setup(out):
    out=Path(out).resolve();out.relative_to(ROOT/'outputs/train40_practical_calibration')
    if (out/'PRACTICAL_STUDY.json').exists():return out
    if out.exists() and any(out.iterdir()):raise FileExistsError('New practical study directory required')
    gate=read(BASE/'SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION.json')
    assert gate['SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION']=='YES' and all(v['passed'] for v in gate['conditions'].values())
    alias=read(BASE/'PAPER_METHOD_ALIAS.json');original={Path(x['path']).name:x for x in alias['workspace_code_manifest']}
    for name in FIXED_BACKEND:assert record(ROOT/'tools/contact_coordination'/name)==original[name],name
    out.mkdir(parents=True,exist_ok=True);stage_inputs(BASE,out)
    ids=read(out/'SPLIT_CONTRACT.json')['authorized_training_source_ids'];assert len(ids)==len(set(ids))==40
    fixed=read(out/'COVERAGE8.json');physical=[fixed['golden'],*fixed['source_ids']]
    # Four additional source-geometry extremes, predeclared without plan/physics.
    geometry={sid:np.asarray(read(out/'source_phase'/sid/'PHASE_RECORD.json')['initial_object_pose_world'])[:3,3] for sid in ids}
    for _ in range(4):
        sid=max((sid for sid in ids if sid not in physical),key=lambda s:(min(np.linalg.norm(geometry[s]-geometry[t]) for t in physical),s))
        physical.append(sid)
    rules=dict(meaningful_improvement=dict(minimum_average_net_complete_gain=2.,minimum_distinct_new_sources=2,
        weaker_method_coverage_is_tie_breaker_only=True),planning_score='0.5*B_complete/40 + 0.5*C_complete/40',
        tie_breakers=['higher weaker-method coverage','fewer NO_IK','fewer NO_COMPLETE_CHAIN','lower mean source deviation','lower planning time','lower recipe complexity'],
        top2_rule='Include rank2 only when average complete count is within0.5 of rank1 and weaker-method count is within1; otherwise top1 only',
        final_selection=['paired mean full-task success/all predeclared physical sources','mean TRAIN40 planning coverage','handoff plus ownership success','lower source deviation','lower pre-acquisition contact displacement','lower planning time','lower complexity'],
        select_C_minus_B_gap=False,new_recipes_after_sweep_forbidden=True)
    from dataclasses import asdict
    from .joint_path_planner import Budget
    budget=asdict(Budget())
    study=dict(schema='train40_practical_v1',created_at=now(),base_run=str(BASE),train_source_ids=ids,
        methods=list(METHODS),recipes=list(RECIPES),physical_source_ids=physical,selection_rules=rules,
        fixed_concurrent_planning_workers=4,concurrent_physx_workers=1,planner_budget=budget,
        candidate_budget=dict(grasp_charts=5,retained_complete_grasp_chains=2,retained_handoff_chains=2,
            connected_IK_endpoints=2,retained_RRT_solutions=4),DEV35_started=False,ACT_started=False,
        scope='Single handoff-place TRAIN calibration and final A/C evaluation; no architecture redesign')
    atomic_json(out/'PRACTICAL_STUDY.json',study)
    atomic_json(out/'CALIBRATION_SEARCH_SPACE.json',dict(default=RECIPES['CURRENT_DEFAULT'],recipes=RECIPES,
        dimensions=6,bounds=dict(BOUNDS,acquisition_posture_multiplier=(.25,1.)),selection_rules=rules,
        architecture_changes_permitted=False,explicitly_authorized_extension='Bounded incidental intended-hand contact and exposure of existing target/IK scalar quantities',
        fixed_planner_budget=budget,fixed_candidate_budget=study['candidate_budget'],all_non_acquisition_legacy_multipliers=1.))
    atomic_json(out/'PHYSICAL_SUBSET_PREDECLARATION.json',dict(source_ids=physical,selected_at=now(),
        rule='Golden plus unchanged Coverage8 plus4 farthest-point initial-object-position TRAIN sources; selected before any new planning/physics outcomes',
        positions=geometry,selection_uses_new_outcomes=False))
    atomic_json(out/'TRAIN40_SOURCE_MANIFEST.json',dict(source_ids=ids,source_count=40,
        evidence=[record(out/'source_phase'/sid/'PHASE_RECORD.json') for sid in ids],DEV35_loaded=False))
    common=dict(schema='train40_practical_calibration',nominal_settle_observation_s=2.,methods=['A_WRIST',*METHODS],
        CALIBRATION_READY=True,FINAL_CONFIG_FROZEN=False,DEV35_started=False,ACT_started=False)
    atomic_json(out/'ABC_STUDY.json',common)
    ready=dict(CALIBRATION_READY=True,checks=dict(source_guided_architecture_verified=True,explicit_TRAIN40_authorization=True),
        evidence=[record(BASE/'SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION.json'),record(out/'PRACTICAL_STUDY.json')])
    atomic_json(out/'CALIBRATION_READY.json',ready)
    from .calibration_parameters import DEFAULT as legacy
    for name,values in RECIPES.items():
        area=out/'recipes'/name;stage_inputs(out,area)
        atomic_json(area/'ABC_STUDY.json',common);atomic_json(area/'CALIBRATION_READY.json',ready)
        atomic_json(area/'CALIBRATION_PARAMETERS.json',dict(values=dict(legacy,acquisition_posture_multiplier=values['acquisition_posture_multiplier'])))
        atomic_json(area/'PRACTICAL_PARAMETERS.json',dict(recipe=name,values={k:values[k] for k in DEFAULT}))
    unchanged=[record(ROOT/'tools/contact_coordination'/n) for n in FIXED_BACKEND]
    atomic_json(out/'ARCHITECTURE_BASELINE_MANIFEST.json',dict(base_alias=record(BASE/'PAPER_METHOD_ALIAS.json'),
        frozen_original_code_manifest=alias['workspace_code_manifest'],unchanged_backend=unchanged,
        base_gate=record(BASE/'SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION.json'),
        allowed_changes='Only scalar parameter plumbing, scoped incidental-contact policy, read-only contact-position telemetry, orchestration/cache dependencies'))
    atomic_text(out/'DECISIONS.md','# Practical TRAIN40 calibration decisions\n\nExactly four recipes and one shared selection rule are predeclared. No DEV35 or ACT. The original repaired run remains unchanged.\n\n'
        'The source-guided architecture has no learned TRAIN-distribution fit to refit: task normalization is per-source task-axis registration and target embodiment calibration is fixed. All40 source evidence is used unchanged; a complete TRAIN40 source-geometry summary will accompany the final freeze.\n')
    atomic_text(out/'RUN_LOG.jsonl','')
    return out


def assert_backend(out):
    manifest=read(Path(out)/'ARCHITECTURE_BASELINE_MANIFEST.json')
    repairs=manifest.get('verified_software_consistency_repairs',[])
    approved={}
    for repair in repairs:
        before,after=repair['before'],repair['after']
        evidence=repair['reproduction_evidence'];regression=repair['regression']
        if (before['path']!=after['path'] or before['path'] in approved or
            before not in manifest['unchanged_backend'] or not evidence or not repair.get('reason') or
            not repair.get('authorization') or not all(record(x['path'])==x for x in [*evidence,regression]) or
            read(regression['path']).get('status')!='PASS' or
            read(regression['path']).get('repaired_backend')!=after):
            raise RuntimeError('INVALID_SOFTWARE_CONSISTENCY_REPAIR: '+before['path'])
        approved[before['path']]=after
    for x in manifest['unchanged_backend']:
        expected=approved.get(x['path'],x)
        if record(x['path'])!=expected:raise RuntimeError('FROZEN_ARCHITECTURE_CHANGED: '+x['path'])
    return dict(backend_bytes_unchanged=not repairs,verified_software_consistency_repairs=repairs,
        unchanged_backend_files=len(manifest['unchanged_backend'])-len(repairs))
