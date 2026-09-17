"""Evidence-backed current-study accounting; unavailable ACT stays unavailable."""
from pathlib import Path
from collections import Counter
import csv
import io
from .io import read, record, atomic_json, atomic_text
from .study_statistics import summarize_policy, paired_result, exact_interval

STAGES = ['LEFT_GRASP', 'LIFT', 'HANDOFF', 'RIGHT_OWNERSHIP', 'RIGHT_TRANSPORT',
          'BIN_ENTRY', 'BIN_SETTLE', 'FULL_TASK']


def csv_table(path, rows):
    if not rows:
        raise ValueError('Explicit scheduled rows are required, even when unexecuted')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def markdown_table(rows):
    keys = list(rows[0])
    def cell(value):
        return 'NOT MEASURED' if value is None else str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |'] +
                     ['| ' + ' | '.join(cell(row.get(k)) for k in keys) + ' |' for row in rows]) + '\n'


def conversion_summary(rows, scheduled):
    physical = [r for r in rows if r.get('physical_run')]
    valid = [r for r in physical if r.get('physical_validity')]
    successes = sum(r.get('task_success') is True for r in valid)
    plans=sum(bool(r.get('full_task_plan')) for r in rows)
    unknown=sum(r.get('terminal') == 'INFRASTRUCTURE_INVALID' for r in rows)
    resolved=len(rows)==scheduled and unknown==0
    stage_keys=[{'LEFT_GRASP':'GRASP','RIGHT_TRANSPORT':'TRANSPORT'}.get(s,s) for s in STAGES]
    cumulative={s:sum(all(r.get('stages',{}).get(k,False) for k in stage_keys[:i+1]) for r in valid)
                for i,s in enumerate(STAGES)}
    return dict(scheduled=scheduled, recorded=len(rows),
                complete_valid_plans=plans,
                plan_construction_rate=plans/scheduled if resolved else None,
                plan_construction_interval95=exact_interval(plans,scheduled) if resolved else None,
                physically_run=len(physical), valid_physical_rollouts=len(valid),
                full_task_successes=successes,
                complete_valid_supervision=sum(bool(r.get('complete_supervision')) for r in rows),
                infrastructure_unknown=unknown,
                not_attempted=scheduled-len(rows),
                cumulative_pipeline_completion=successes/scheduled if resolved else None,
                physical_completion=successes/len(valid) if valid else None,
                physical_completion_interval95=exact_interval(successes, len(valid)),
                cumulative_valid_stage_counts=cumulative,
                first_failures=dict(Counter(str(r.get('first_failure')) for r in rows)))


def conversion_rows(schedule, observed):
    keyed = {(r['source_id'], r['condition']): r for r in observed}
    if len(keyed) != len(observed):
        raise ValueError('Duplicate conversion outcome')
    result = []
    for item in schedule:
        r = keyed.get((item['source_id'], item['condition']))
        result.append(dict(index=item['index'], source_id=item['source_id'], condition=item['condition'],
            attempted=r is not None, terminal=r['terminal'] if r else 'NOT_ATTEMPTED_UPSTREAM',
            full_task_plan=bool(r and r.get('full_task_plan')),
            physically_run=bool(r and r.get('physical_run')),
            physical_validity=r.get('physical_validity') if r else None,
            complete_supervision=bool(r and r.get('complete_supervision')),
            full_task_success=r.get('task_success') if r else None,
            first_failure=r.get('first_failure') if r else 'UPSTREAM_PREREQUISITE',
            planning_wall_s=r.get('planning_process_wall_s') if r else None,
            context=r.get('context') if r else None,
            **{s: r.get('stages', {}).get({'LEFT_GRASP':'GRASP', 'RIGHT_TRANSPORT':'TRANSPORT'}.get(s,s))
               if r and r.get('physical_run') else 'NOT_ATTEMPTED' for s in STAGES}))
    return result


def audit_zero_supervision(out):
    """Audit only after every frozen attempt; never manufacture a fallback pair."""
    from .generation_study import verify_freeze, matched_manifest
    from .target_dataset import checked_episode
    verify_freeze(out)
    counts = matched_manifest(out)
    if counts['paired_usable']:
        return dict(status='NONEMPTY_MATCHED_SET_CONTINUE_ACT', counts=counts)
    ledger = read(out/'TRAIN40_conversion/LEDGER.json')
    if ledger['completed'] != 80:
        raise ValueError('Generation is not finished')
    diagnostic = read(out/'dataset_interface_diagnostic/RESULT.json')
    if diagnostic['status'] != 'REAL_G1_TRACE_DATASET_INTERFACE_VERIFIED':
        raise ValueError('Real observation/action serialization diagnostic is required')
    for dep in diagnostic['dataset_files']:
        if record(dep['path']) != dep:
            raise ValueError('Changed dataset-interface diagnostic')
    rows = []
    for r in ledger['rows']:
        for dep in r.get('artifacts', []):
            if record(dep['path']) != dep:
                raise ValueError('Changed conversion evidence')
        if r.get('complete_supervision'):
            checked_episode(r)
        alignment_summary=None
        if r.get('physical_run'):
            alignment_record=r['alignment']
            if record(alignment_record['path'])!=alignment_record:
                raise ValueError('Changed physical observation/action alignment evidence')
            alignment=read(alignment_record['path'])
            if alignment['status']!='DYNAMIC_OBSERVATION_ACTION_ALIGNMENT_VERIFIED':
                raise ValueError('Diagnose the physical observation/action interface before accepting zero supervision')
            eligible=bool(alignment['physical_validity'] and alignment['complete_command_sequence'])
            if eligible!=bool(r.get('complete_supervision')):
                raise ValueError('Supervision eligibility disagrees with measured alignment evidence')
            alignment_summary={key:alignment[key] for key in ('frames','complete_command_sequence','physical_validity',
                'maximum_action_alignment_error_rad','maximum_state_alignment_error_rad')}
        if r['terminal'] == 'INFRASTRUCTURE_INVALID':
            raise ValueError('Diagnose unresolved common infrastructure before accepting zero supervision')
        reason = ('NO_COMPLETE_VALID_PLAN' if not r.get('full_task_plan') else
                  'INVALID_OR_INCOMPLETE_DYNAMIC_EXECUTION' if not r.get('complete_supervision') else
                  'VALID_UNPAIRED_SUPERVISION')
        rows.append(dict(source_id=r['source_id'], condition=r['condition'], terminal=r['terminal'],
                         supervision_missing_reason=reason, first_failure=r.get('first_failure'),
                         context=r.get('context'),physical_alignment=alignment_summary))
    result = dict(status='ZERO_PAIRED_SUPERVISION_AFTER_COMMON_INTERFACE_AUDIT',
                  generation=record(out/'generation_freeze/CONTRACT.json'), ledger=counts['ledger'], counts=counts,
                  real_target_interface_diagnostic=record(out/'dataset_interface_diagnostic/RESULT.json'),
                  rows=rows, stop_condition=3,
                  conclusion='The frozen attempts provide no nonempty paired complete valid supervision. Real G1 RGB/state/action serialization and installed ACT chunk loading work on the separate verified development trace. No unresolved common dataset/API error was observed in the recorded conversion outcomes. This does not prove global geometric infeasibility. Changing converter search behavior requires a new study version, not hidden rescue of these outcomes.',
                  legacy_checkpoint_substitution_allowed=False, ACT_results='NOT_MEASURED')
    atomic_json(out/'matched_datasets/ZERO_PAIRED_INTERFACE_AUDIT.json', result)
    csv_table(out/'matched_datasets/SUPERVISION_MISSING_REASONS.csv', rows)
    return result


def saved_failure_details(out, observed):
    """Read rejected phase candidates; never rerun IK or create a checker."""
    details=[]
    for row in observed:
        if row.get('full_task_plan'):
            continue
        context=Path(row['context'])
        paths=sorted(context.glob('prototype/*/morphology_acquisition_v4/PLAN_RESULT.json'))
        paths+=sorted(context.glob('prototype/*/full_task_connection/*/contact_*/PHASE_IK.json'))
        rejected=[]
        for path in paths:
            data=read(path)
            branches=[('final_phase_record',data)]
            branches += [(key,data[key]) for key in ('original_contact_orientation_preparation','free_preparation_retry')
                         if isinstance(data.get(key),dict)]
            for branch,part in branches:
                for phase in part.get('phases',[]):
                    if phase.get('admissible'):
                        continue
                    candidates=[]
                    for candidate in phase.get('candidates',[]):
                        check=candidate.get('connection_validation',{})
                        contacts=check.get('forbidden_contacts',[])
                        candidates.append(dict(seed=candidate.get('seed_index'),
                            goal_satisfied=candidate.get('goal_satisfied'),
                            hard_pose_constraint=candidate.get('hard_pose_constraint'),
                            errors=candidate.get('errors'),q=candidate.get('q'),
                            realized_wrist_pose_world=candidate.get('realized_wrist_pose_world'),
                            connection_valid=check.get('valid'),connection_samples=check.get('samples'),
                            finger_prediction=check.get('finger_prediction'),contact_mode=check.get('contact_mode'),
                            allowed_object_digits=check.get('allowed_object_digits'),
                            forbidden_count=check.get('forbidden_count'),
                            first_forbidden_contact=contacts[0] if contacts else None,
                            max_recorded_forbidden_depth_m=max((c.get('depth_m',0) for c in contacts),default=None)))
                    rejected.append(dict(evidence=record(path),branch=branch,phase=phase.get('phase'),
                        selected_seed=phase.get('selected_seed'),selected_errors=phase.get('selected_errors'),
                        candidates=candidates))
        details.append(dict(source_id=row['source_id'],condition=row['condition'],
            first_failure=row.get('first_failure'),terminal=row['terminal'],context=str(context),
            rejected_phase_records=rejected,
            final_command_validation=[dict(evidence=record(path),
                first_forbidden_contact=(read(path).get('forbidden') or [None])[0],
                forbidden_count=len(read(path).get('forbidden',[])),
                max_recorded_forbidden_depth_m=max((c.get('depth_m',0) for c in read(path).get('forbidden',[])),default=None),
                command_limit_excess_rad=read(path).get('command_limit_excess_rad'))
                for path in sorted(context.glob('prototype/*/full_task_connection/*/physical_plan/FINAL_COMMAND_VALIDATION.json'))],
            interpretation='Saved bounded-search diagnostics, including unsuccessful internal alternatives. Contact sample indices refer to the recorded connecting-edge checks, which include endpoints. Missing fields are unknown, not evidence of safety. A wall-budget termination can precede a final phase record. No global reachability conclusion.'))
    result=dict(implementation=record(__file__),recorded_instances=len(observed),rows=details,
        runtime_physical_evidence=False,read_only_extraction=True)
    atomic_json(out/'SAVED_REJECTED_PHASE_DETAILS.json',result)
    return result


def analyze(out):
    from .generation_study import verify_freeze
    frozen = verify_freeze(out)
    split = read(out/'SPLIT_CONTRACT.json')
    generation = read(out/'TRAIN40_conversion/LEDGER.json')
    policy_path = out/'ACT_DEV35/LEDGER.json'
    policy = read(policy_path) if policy_path.exists() else dict(rows=[], completed=0)
    for row in policy['rows']:
        for key in ('score', 'process'):
            if key in row and record(row[key]['path']) != row[key]:
                raise ValueError('Changed ACT outcome evidence')
    primary = summarize_policy(policy['rows'], split['evaluation_source_ids'])
    counts_path = out/'matched_datasets/CONVERSION_COUNTS.json'
    counts = read(counts_path) if counts_path.exists() else None
    converters = {c: conversion_summary([r for r in generation['rows'] if r['condition']==c], 40) for c in ('A', 'B')}
    reference_contract = read(out/'reference_coupling10/CONTRACT.json')
    reference_path = out/'reference_coupling10/LEDGER.json'
    reference = read(reference_path) if reference_path.exists() else dict(rows=[], completed=0)
    for row in generation['rows']+reference['rows']:
        dependencies=list(row.get('artifacts',[]))
        if row.get('planning_log'):
            dependencies.append(row['planning_log'])
        if row.get('score'):
            score=read(row['score']['path'])
            dependencies.append(score['trace'])
        for dep in dependencies:
            if record(dep['path'])!=dep:
                raise ValueError('Changed conversion/reference outcome evidence')
    physical_details=[]
    for row in generation['rows']+reference['rows']:
        if not row.get('physical_run'):
            continue
        folder=Path(row['attempt'])
        parts={name:read(folder/name) for name in ('LEGACY_CONTROL_SCORER.json','MEASURED_GEOMETRY_AUDIT.json','PHASE_STOP.json')
               if (folder/name).exists()}
        legacy=parts.get('LEGACY_CONTROL_SCORER.json',{})
        physical_details.append(dict(source_id=row['source_id'],condition=row['condition'],scope=row['scope'],
            physical_validity=row['physical_validity'],first_task_failure=row.get('first_failure'),
            numerical_motion_validity=legacy.get('physical_validity'),integrity=legacy.get('integrity'),
            measured_state_geometry_audit=parts.get('MEASURED_GEOMETRY_AUDIT.json'),
            actual_phase_stop=parts.get('PHASE_STOP.json'),
            evidence=[record(folder/name) for name in parts],
            interpretation='Raw numerical validity and measured-state collider overlap are separate checks. Geometry overlap is reconstructed from measured articulation and runtime-compatible colliders; it is not a fabricated contact-force measurement. Raw instantaneous velocity is not necessarily an interval pose derivative.'))
    atomic_json(out/'FROZEN_PHYSICAL_VALIDITY_AUDIT.json',dict(rows=physical_details,implementation=record(__file__)))
    saved_failure_details(out,generation['rows']+reference['rows'])
    ref_summary = {c: conversion_summary([r for r in reference['rows'] if r['condition']==c], 10) for c in ('B', 'B_NO_COUPLING')}
    geometry_rows=[]
    for row in reference['rows']:
        pointer=Path(row['context'])/'target_repair/CURRENT_CONTACT_REGION_FIT.json'
        if not pointer.exists():continue
        enabled=row['condition']=='B'
        bank=Path(read(pointer)['path'])/f'coupling_{enabled}.json'
        if not bank.exists():continue
        for candidate in read(bank):
            geometry_rows.append(dict(source_id=row['source_id'],condition=row['condition'],
                contact_candidate=candidate['contact_candidate'],seed=candidate['seed'],
                position_disagreement_mm=1000*candidate['position_disagreement_m'],
                rotation_disagreement_rad=candidate['rotation_disagreement_rad'],
                valid_stationary_overlap=candidate['valid_stationary_overlap'],
                unary_and_enabled_factor_cost=candidate['cost'],bank_path=str(bank),bank_sha256=record(bank)['sha256'],
                physical_measurement=False))
    ref_paired = dict(status='NOT_ESTIMABLE_UNTIL_REFERENCE_ATTEMPTS_RECORDED', paired_N=0)
    if reference['completed'] == 20 and not any(r['terminal']=='INFRASTRUCTURE_INVALID' for r in reference['rows']):
        rr = {(r['source_id'], r['condition']): r for r in reference['rows']}
        ref_paired = paired_result([rr[s,'B_NO_COUPLING'].get('task_success') is True for s in reference_contract['source_ids']],
                                   [rr[s,'B'].get('task_success') is True for s in reference_contract['source_ids']])
        ref_paired.update(interpretation='Reference pipeline completion on all ten scheduled source pairs, including no-plan outcomes; not conditional physical completion and not ACT coupling attribution.',
                          B_label='INTERACTION_OURS', A_label='OURS_NO_COUPLING',
                          paired_physical_N=sum(all(rr[s,c].get('physical_validity') for c in ('B','B_NO_COUPLING')) for s in reference_contract['source_ids']))
    result = dict(status='VERIFIED_CURRENT_NUMERIC_ACCOUNTING', generation_recorded=generation['completed'],
                  converters=converters, matched_counts=counts, ACT=primary,
                  reference_recorded=reference['completed'], reference=ref_summary, reference_paired=ref_paired,
                  reference_geometry=geometry_rows,
                  evidence=[record(out/'TRAIN40_conversion/LEDGER.json'), record(out/'generation_freeze/CONTRACT.json')],
                  implementation=record(__file__))
    for p in (policy_path, reference_path, counts_path):
        if p.exists(): result['evidence'].append(record(p))
    atomic_json(out/'PAPER_NUMERIC_SUMMARY.json', result)
    csv_table(out/'TRAIN40_PER_INSTANCE_RESULTS.csv', conversion_rows(frozen['schedule'], generation['rows']))
    converter_table = [dict(condition=c, TRAIN_sources=40, **{k:v for k,v in converters[c].items() if k not in ('first_failures','physical_completion_interval95')}) for c in ('A','B')]
    csv_table(out/'TABLE_CONVERTER_RESULTS.csv', converter_table)
    readable_conversion=[]
    for c in ('A','B'):
        s=converters[c]
        readable_conversion.append(dict(condition=c,scheduled=40,plans=s['complete_valid_plans'],
            physical_run=s['physically_run'],valid_physical=s['valid_physical_rollouts'],
            full_task_pipeline=f"{s['full_task_successes']}/40",
            conditional_physical=(f"{s['full_task_successes']}/{s['valid_physical_rollouts']}" if s['valid_physical_rollouts'] else 'NOT MEASURED'),
            usable_supervision=s['complete_valid_supervision'],infrastructure_unknown=s['infrastructure_unknown']))
    atomic_text(out/'TABLE_CONVERTER_RESULTS.md', markdown_table(readable_conversion)+
        '\nTRAIN40 frozen conversion, not ACT evaluation. Exact rate intervals and cumulative valid stage counts are in the CSV and PAPER_NUMERIC_SUMMARY.json. '+
        'Complete valid task failures can provide supervision but are labeled non-expert. Missing/invalid/incomplete traces are excluded symmetrically.\n')
    act_table = [dict(policy='ACT-'+c, **{k:v for k,v in primary['conditions'][c].items() if k not in ('cumulative_valid_stage_counts','full_task_interval95')}) for c in ('A','B')]
    csv_table(out/'TABLE_ACT_DEV35_RESULTS.csv', act_table)
    readable_act=[]
    for c in ('A','B'):
        s=primary['conditions'][c]
        readable_act.append(dict(policy='ACT-'+c,scheduled=35,recorded=s['recorded'],
            valid_physical=s['valid_physical_rollouts'],full_task=(f"{s['full_task_successes']}/35" if s['full_task_rate'] is not None else s['primary_status']),
            not_attempted=s['not_attempted'],infrastructure_unknown=s['infrastructure_unknown']))
    atomic_text(out/'TABLE_ACT_DEV35_RESULTS.md', markdown_table(readable_act)+
        '\nDEV35 is development evaluation. Unexecuted ACT has NOT MEASURED TSR, not 0%. Reference outcomes never enter this table. '+
        'B−A and paired uncertainty are in PAPER_NUMERIC_SUMMARY.json; they remain unestimable without resolved matched policy outcomes.\n')
    csv_table(out/'PER_EPISODE_ACT_RESULTS.csv', primary['per_source'])
    reference_rows = conversion_rows(reference_contract['schedule'], reference['rows'])
    csv_table(out/'TABLE_COUPLING_ABLATION.csv', reference_rows)
    if geometry_rows:csv_table(out/'REFERENCE_COUPLING_CANDIDATE_GEOMETRY.csv',geometry_rows)
    atomic_text(out/'TABLE_COUPLING_ABLATION.md', markdown_table([{k:r[k] for k in ('source_id','condition','terminal','full_task_plan','physically_run','full_task_success','first_failure')} for r in reference_rows])+'\nExploratory reference-level paired10. No ACT-no-coupling policy is trained.\n')
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--zero-audit', action='store_true')
    a = p.parse_args()
    print((audit_zero_supervision(a.run_dir) if a.zero_audit else analyze(a.run_dir))['status'])
