"""Thin frozen TRAIN40 scheduling and matched-supervision accounting."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time
from .io import ROOT,OFFLINE,read,record,atomic_json,atomic_text,fingerprint


def verify_freeze(out):
    frozen=read(out/'generation_freeze/CONTRACT.json')
    if frozen['status']!='FROZEN_TRAIN40_GENERATION':raise ValueError('No generation freeze')
    for item in frozen['dependencies']:
        if record(item['path'])!=item:raise ValueError('Generation dependency changed: '+item['path'])
    return frozen


def freeze(out):
    from .generalization_gate import require
    require(out, 'generation_freeze')
    path=out/'generation_freeze/CONTRACT.json'
    if path.exists():return verify_freeze(out)
    from .demo_alignment import scored_task_success
    m=read(out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json');score=read(m['score']['path'])
    assert record(m['score']['path'])==m['score'] and record(score['trace']['path'])==score['trace']
    if not scored_task_success(score):raise ValueError('Source full-task prerequisite unverified')
    pilot=read(out/'TRAIN_pilot/PILOT_LEDGER.json')
    if pilot['scheduled_instances']!=10 or len(pilot['rows'])!=10:raise ValueError('Fixed pilot incomplete')
    split=read(out/'SPLIT_CONTRACT.json');ids=split['authorized_training_source_ids']
    assert len(ids)==40 and len(set(ids))==40 and not set(ids)&set(split['evaluation_source_ids'])
    module_names=['io.py','source_phase.py','source_contract.py','targets.py','prototype.py','morphology_repair.py',
        'wrist_reference.py','episode_context.py','acquisition_plan.py','handoff_repair.py','full_task_plan.py','planner.py',
        'giver_clearance.py','loaded_contact_geometry.py','runtime_hulls.py','physical_attempt.py','phase_clock_runtime.py',
        'phase_physics.py','demonstration_physics.py','demonstration_observation.py','g1_rgb_observer.py','calibration_capture.py',
        'score_hybrid.py','demo_alignment.py','conversion_attempt.py']
    module_names += ['receiving_relation.py','scientific_cache.py','planning_kinematics.py','generation_study.py','contact_transition_geometry.py',
        'contact_candidate_region.py','passive_hand_clearance.py','generalization_gate.py','targeted_batch_worker.py']
    paths=[ROOT/'tools/contact_coordination'/n for n in module_names]
    paths += [ROOT/'tools/doll_handoff_retargeting'/n for n in ('models.py','common.py')]
    paths += [ROOT/'tools'/n for n in ('finalize_common_dex3_grasp_qualification.py','deployment_camera_config.py',
        'direct_physical_execution_layer.py','common_execution_layer.py','common_execution_isaac_runtime.py')]
    paths += [Path(d['path']) for d in read(ROOT/'outputs/contact_coordination_hybrid/20260907T073437Z/common_control/scripted_captured/DEPENDENCIES.json')['files']]
    paths += [out/'target_repair'/n for n in ('CONTACT_CALIBRATION.json','CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json',
        'LOADED_CARRY_GEOMETRY_f83d2e29156f.json','CONTACT_SEPARATION_CANDIDATES.json','CALIBRATED_HANDOFF_GRAVITY.json',
        'CALIBRATED_RIGHT_CARRY_GRAVITY.json','runtime_bin150/RUNTIME_HULLS.json','runtime_bin150/RUNTIME_HULL_VERTICES.npz')]
    paths += [out/'SPLIT_CONTRACT.json',out/'bootstrap/SELECTION.json',out/'TRAIN_pilot/PILOT_LEDGER.json',
        ROOT/'outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json',
        Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml'),
        Path('/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd'),
        out/'act_training/SUPERVISION_ELIGIBILITY_PREDECLARATION.json',ROOT/'outputs/policy_b_isaac_validation/camera/source_like_cam_high.json',
        ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json',
        ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json']
    if (out/'golden_batch/CURRENT_PHYSICAL_EQUIVALENCE.json').exists():paths.append(out/'golden_batch/CURRENT_PHYSICAL_EQUIVALENCE.json')
    paths += [out/'GENERALIZATION_REPAIR_READY.json',out/'target_repair/TASK_FRAME_NORMALIZATION.json']
    for sid in ids:
        paths += [p for p in (out/'source_phase'/sid).iterdir() if p.suffix in ('.json','.npz')]
    signature,dependencies=fingerprint(paths)
    schedule=[]
    for index,sid in enumerate(ids):
        for condition in (('A','B') if index%2==0 else ('B','A')):
            schedule.append(dict(index=len(schedule),source_id=sid,condition=condition,physical_primary_attempts=1))
    result=dict(status='FROZEN_TRAIN40_GENERATION',version=1,created_unix_s=time.time(),dependencies=dependencies,
        scientific_signature=signature,source_ids=ids,schedule=schedule,scheduled_instances=80,
        planning_wall_budget_s=900,physics_timeout_s=1200,identical_infrastructure_retry_limit=3,
        budget_basis='Unchanged900second common wall cap includes acquisition, both coupling fits, candidate selection and final command validation. Current measured TRAIN planning times are preserved in the prototype and pilot receipts; no desired success rate or DEV outcome determines this cap.',
        solver_budget=dict(phase_seeds=3,phase_nfev_per_seed=120,collision_refinement_seeds=1,collision_refinement_nfev=120,
            contact_fit_seeds=2,contact_fit_nfev=180,contact_candidate_count=21,
            original_contact_candidate_fractions=[0.,.5,1.],forced_contact_option=None,
            additional_contact_directions='Object-box face normals and face diagonals inside unchanged calibrated displacement radius',
            passive_hand_preparation_fractions=[.5,1.],passive_hand_preparation_directions=['toward named shoulder','away from source-conditioned carry object','up'],
            transport_region_max=8,placement_region_max=54,release_geometry_refinements_max=3,delayed_release_options_max=3,
            lower_release_region_max=21,cartesian_empty_retreat_knots=6,cartesian_nfev_per_seed=40),
        common_region_hierarchy=dict(screen_all_existing_candidates=True,collision_refinement_top_k=3,
            ranking=['pose satisfaction','collision penetration','IK cost','deterministic candidate index'],
            accepted_paths_receive_complete_validation=True),
        cache_accounting='Only exact scientific/input compatible complete stages can resume. Retain original charged plan time. Pilot/prototype execution outcomes are not substituted for these frozen TRAIN40 attempts.',
        task_instruction='Pick up the doll with the left hand, pass it to the right hand, and place it in the bin.',
        supervision='Complete valid dynamic command sequence with all pre-action target G1 RGB/measured28/action28 pairs. Complete valid task failures retain their outcome label and are not labeled successful experts. Incomplete/unsafe/diagnostic traces excluded symmetrically.',
        paired_membership='Ordered intersection of complete valid aligned A and B sources; smaller nonempty matched membership explicitly authorized by the latest user instruction.',
        checkpoint_selection='Predeclared equal fixed final100000training step. DEV35 never used for checkpoint selection.',
        prototype_milestone=record(out/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json'),pilot=record(out/'TRAIN_pilot/PILOT_LEDGER.json'),
        observed_pilot=dict(A_plans=sum(bool(r.get('full_task_plan')) for r in pilot['rows'] if r['condition']=='A'),
            B_plans=sum(bool(r.get('full_task_plan')) for r in pilot['rows'] if r['condition']=='B'),
            physical_runs=sum(bool(r.get('physical_run')) for r in pilot['rows']),
            valid_physical_runs=sum(bool(r.get('physical_validity')) for r in pilot['rows'])),
        generalization_repair_evidence=record(out/'GENERALIZATION_REPAIR_READY.json'),
        threshold_or_success_rate_gate=False,DEV_outcomes_read=False)
    atomic_json(path,result)
    atomic_text(out/'generation_freeze/README.md','Frozen TRAIN40 generation uses the exact dependencies and80-instance schedule in CONTRACT.json. The pilot was weak; no success quota was applied. Later changes to scientific dependencies require a new version and symmetric affected reruns.\n')
    return result


def run(out,resume=False):
    from .generalization_gate import require
    require(out, 'matched_dataset_realization')
    frozen=verify_freeze(out);area=out/'TRAIN40_conversion';area.mkdir(exist_ok=True)
    ledger_path=area/'LEDGER.json';rows=read(ledger_path)['rows'] if ledger_path.exists() else []
    if rows and not resume:raise ValueError('Existing frozen generation: use --resume')
    from .episode_context import prepare
    from .conversion_attempt import attempt
    with (area/'.generation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in frozen['schedule']:
            if any(r['index']==item['index'] for r in rows):continue
            verify_freeze(out)
            sid,condition=item['source_id'],item['condition'];context=prepare(out,sid,condition,'TRAIN40_conversion')
            args=[OFFLINE,'-m','tools.contact_coordination.conversion_attempt','--run-dir',str(out),'--source-id',sid,'--condition',condition,'--scope','TRAIN40_conversion','--plan-only','--resume']
            log=area/'logs'/f"{item['index']:03d}_{condition}_{sid}.log";log.parent.mkdir(exist_ok=True)
            started=time.monotonic();timeout=False
            print('TRAIN40_START',item['index']+1,80,condition,sid,flush=True)
            with log.open('a') as stream:
                process=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                atomic_json(area/'ACTIVE_PROCESS.json',dict(pid=process.pid,command=args,index=item['index'],phase='planning',started_unix_s=time.time()))
                try:code=process.wait(timeout=frozen['planning_wall_budget_s'])
                except subprocess.TimeoutExpired:
                    timeout=True;process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    code='FIXED_PLANNING_BUDGET_EXHAUSTED'
            planning_elapsed=time.monotonic()-started
            if timeout:
                row=dict(item,terminal='NO_PLAN_WITHIN_FIXED_BUDGET',first_failure='PLANNING_WALL_BUDGET',full_task_plan=False,physical_run=False,complete_supervision=False,task_success=None,context=str(context))
            elif code!=0:
                atomic_json(area/'INFRASTRUCTURE_FAILURE.json',dict(item=item,returncode=code,log=record(log),context=str(context)))
                raise RuntimeError('Diagnose and preserve the implementation failure before resume: '+str(log))
            else:
                row=attempt(out,sid,condition,'TRAIN40_conversion',execute=True,resume=True);row=dict(row,**item)
            row.update(planning_process_wall_s=planning_elapsed,planning_log=record(log),generation_contract=record(out/'generation_freeze/CONTRACT.json'))
            rows.append(row);rows.sort(key=lambda r:r['index'])
            result=dict(status='FROZEN_TRAIN40_COMPLETE' if len(rows)==80 else 'FROZEN_TRAIN40_RUNNING',scheduled=80,completed=len(rows),rows=rows)
            atomic_json(ledger_path,result)
            with (out/'RUN_LOG.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=time.time(),stage='matched_dataset_realization',completed=len(rows),scheduled=80,condition=condition,source_id=sid,terminal=row['terminal']))+'\n')
            print('TRAIN40_DONE',len(rows),80,condition,sid,row['terminal'],flush=True)
        # An already completed resume does not execute the loop body.
        return read(ledger_path)


def matched_manifest(out):
    frozen=verify_freeze(out);ledger=read(out/'TRAIN40_conversion/LEDGER.json')
    if ledger['completed']!=80:raise ValueError('All TRAIN40 source/condition attempts must be recorded first')
    eligible={c:{} for c in ('A','B')}
    from .target_dataset import checked_episode
    for row in ledger['rows']:
        if row.get('complete_supervision'):
            checked_episode(row);eligible[row['condition']][row['source_id']]=row
    paired=[sid for sid in frozen['source_ids'] if all(sid in eligible[c] for c in eligible)]
    summary=dict(TRAIN_sources=40,A_valid=len(eligible['A']),B_valid=len(eligible['B']),paired_usable=len(paired),paired_source_ids=paired,
        ledger=record(out/'TRAIN40_conversion/LEDGER.json'),eligibility_rule=frozen['supervision'],zero_is_not_an_authorized_fabrication_or_source_substitution=True)
    atomic_json(out/'matched_datasets/CONVERSION_COUNTS.json',summary)
    if paired:
        manifest=dict(status='FROZEN_MATCHED_TARGET_SUPERVISION',paired_source_ids=paired,task_instruction=frozen['task_instruction'],
            conditions={c:[eligible[c][sid] for sid in paired] for c in eligible},generation=record(out/'generation_freeze/CONTRACT.json'),conversion_counts=record(out/'matched_datasets/CONVERSION_COUNTS.json'))
        atomic_json(out/'matched_datasets/MEMBERSHIP.json',manifest)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--freeze',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--membership',action='store_true')
    a=p.parse_args();print(freeze(a.run_dir) if a.freeze else matched_manifest(a.run_dir) if a.membership else run(a.run_dir,a.resume))
