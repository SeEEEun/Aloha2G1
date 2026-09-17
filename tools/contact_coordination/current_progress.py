"""Disk-backed progress for the authorized TRAIN40 continuation."""
from pathlib import Path
import numpy as np
from .io import read, record, atomic_json


def status(out):
    split = read(out / 'SPLIT_CONTRACT.json')
    assert split['status'] == 'AUTHORIZED_TRAIN40_DEV35'
    assert split['authorized_training_count'] == 40 and not split['train_eval_source_overlap']
    assert record(split['manifest']['path']) == split['manifest']
    rows = []
    folders=sorted([*(out/'prototype').glob('*/handoff_prefix/*/physics_attempt_*'),
                    *(out/'prototype').glob('*/full_task_connection/*/physics_attempt_*'),
                    *(out/'TRAIN_pilot/contexts').glob('*/*/*/prototype/*/full_task_connection/*/physics_attempt_*')])
    for folder in folders:
        invocation=read(folder/'INVOCATION.json') if (folder/'INVOCATION.json').exists() else {}
        row = dict(folder=str(folder), full_task=invocation.get('full_task',False))
        if (folder / 'PROCESS.json').exists():
            row['process'] = read(folder / 'PROCESS.json')
            if (folder / 'HYBRID_SCORE.json').exists():
                score = read(folder / 'HYBRID_SCORE.json')
                assert record(score['trace']['path']) == score['trace']
                row.update(score=record(folder / 'HYBRID_SCORE.json'), stages=score['stages'],
                           physical_validity=score['physical_validity'],
                           right_only_opposing_s=score['right_only_opposing_s'])
        elif (folder / 'MEASURED_CHECKPOINT.npz').exists():
            with np.load(folder / 'MEASURED_CHECKPOINT.npz') as trace:
                row['latest_saved_control_frame'] = int(trace['control_frame'][-1])
            row['process_status'] = 'NO_FINAL_PROCESS_RECEIPT_CHECK_ACTIVE_PROCESS_BEFORE_RESUME'
        else:
            row['process_status'] = 'NO_FINAL_PROCESS_RECEIPT'
        rows.append(row)
    milestone = out / 'FIRST_SOURCE_CONDITIONED_FULL_TASK.json'
    # A milestone file is only accepted with a physical score tied to its trace.
    full_task = False
    if milestone.exists():
        evidence = read(milestone)
        score = read(evidence['score']['path'])
        assert record(evidence['score']['path']) == evidence['score']
        assert record(score['trace']['path']) == score['trace']
        from .demo_alignment import scored_task_success
        full_task = scored_task_success(score)
    result = dict(status='TRAIN_DEVELOPMENT_IN_PROGRESS' if not full_task else 'SOURCE_FULL_TASK_VERIFIED_CONTINUE_PILOT',
                  authorized_TRAIN_sources=40, DEV_scenes=35,
                  source_conditioned_full_task=full_task, reference_attempts=rows,
                  handoff_prefix_attempts=[r for r in rows if not r['full_task']],
                  completed_prefix_processes=sum('process' in r and not r['full_task'] for r in rows),
                  completed_natural_start_processes=sum('process' in r for r in rows),
                  primary_ACT_results='NOT_MEASURED' if not (out / 'ACT_FINAL_FREEZE.json').exists() else 'READ_FROZEN_POLICY_LEDGER',
                  training_source_shortfall=0, implementation=record(__file__))
    if (out/'TRAIN40_conversion/LEDGER.json').exists():
        ledger=read(out/'TRAIN40_conversion/LEDGER.json')
        result['generation']=dict(scheduled=80,completed=ledger['completed'],
            planned=sum(bool(r.get('full_task_plan')) for r in ledger['rows']),
            physical_runs=sum(bool(r.get('physical_run')) for r in ledger['rows']),
            complete_supervision=sum(bool(r.get('complete_supervision')) for r in ledger['rows']))
        result['status']='FROZEN_TRAIN40_GENERATION' if ledger['completed']<80 else 'READ_MATCHED_SUPERVISION_COUNTS'
    for name,path in [('matched_training','matched_datasets/CONVERSION_COUNTS.json'),('ACT_training','act_training/ACT_TRAINING_COMPLETE.json'),('ACT_DEV35','ACT_DEV35/LEDGER.json')]:
        if (out/path).exists():result[name]=read(out/path)
    final_path=out/'FINAL_VERIFICATION.json'
    if final_path.exists():
        final=read(final_path)
        dependencies=[final['report'],final['numeric_summary'],*final.get('final_files',[])]
        current=all(record(dep['path'])==dep for dep in dependencies)
        result['final_verification']=record(final_path)
        result['status']=final['status'] if current else 'FINAL_PACKAGE_CHANGED_REVERIFY'
    atomic_json(out / 'CURRENT_PROGRESS.json', result)
    return result


def dispatch(out, stage, resume):
    from .generalization_gate import require
    require(out, stage)
    if stage in ('status', 'inspect_splits'):
        return status(out)
    if stage == 'target_repair' and resume:
        pointer = read(out / 'target_repair/CURRENT_CONTACT_REGION_FIT.json')
        for key in ('result', 'config'):
            assert record(pointer[key]['path']) == pointer[key]
        return dict(status='PERSISTED_CONTACT_FIT_VERIFIED', **pointer)
    if stage == 'prototype':
        if status(out)['source_conditioned_full_task']:return status(out)
        pointer = read(out / 'target_repair/CURRENT_HANDOFF_PREFIX.json')
        plan = Path(pointer['path'])
        attempt = plan.parent / 'physics_attempt_01'
        if not attempt.exists() and not resume:
            from .physical_attempt import launch
            metadata=read(plan/'PLAN.json')
            diagnostic=metadata.get('full_task') is not True or metadata.get('diagnostic',False)
            launch(plan, diagnostic=diagnostic, capture_observations=True)
        if (attempt / 'PROCESS.json').exists() and (attempt / 'event_log.npz').exists():
            if not (attempt / 'HYBRID_SCORE.json').exists():
                from .score_hybrid import score
                score(attempt, out)
            if (attempt / 'OBSERVATION_ACTION.npz').exists() and not (attempt / 'DEMONSTRATION_ALIGNMENT.json').exists():
                from .demo_alignment import validate
                validate(attempt, plan / 'COMMANDS.npz')
        return status(out)
    if stage=='TRAIN_pilot':return read(out/'TRAIN_pilot/PILOT_LEDGER.json')
    if stage=='control':
        from .io import ROOT
        previous=ROOT/'outputs/contact_coordination_hybrid_act/20260907T102755Z/common_control'
        controls=read(previous/'RECAPTURE_RESULTS.json');traces=[]
        for side in ('left','right'):
            folder=previous/(side+'_full_state')
            for dep in read(folder/'DEPENDENCIES.json')['files']:
                if record(dep['path'])!=dep:raise ValueError('Common-control dependency changed')
            control=next(r for r in controls if r['side']==side)
            if control['returncode']!=0 or control['mechanical_score']['status']!='PASS':raise ValueError('Measured common component control failed')
            trace=record(folder/'event_log.npz')
            if trace['sha256']!=control['mechanical_score']['artifact_sha256']['event_log']:raise ValueError('Changed common physical trace')
            traces.append(trace)
        return dict(status='CURRENT_COMPATIBLE_COMMON_CONTROL_REUSED',scope='Bounded bilateral acquisition/lift/retention/natural release; natural complete source task separately verified by M2',result=record(previous/'RECAPTURE_RESULTS.json'),traces=traces)
    if stage=='generation_freeze':
        from .generation_study import freeze
        return freeze(out)
    if stage=='matched_dataset_realization':
        from .generation_study import run,matched_manifest
        run(out,resume);counts=matched_manifest(out)
        if counts['paired_usable']==0:
            import subprocess
            from .io import ROOT
            subprocess.run(['/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python','-m','tools.contact_coordination.dataset_interface_probe',
                '--run-dir',str(out)],check=True,cwd=ROOT)
            from .paper_results import audit_zero_supervision
            return audit_zero_supervision(out)
        import subprocess,os
        from .io import ROOT
        for condition in ('A','B'):
            destination=out/'matched_datasets'/condition
            if (destination.parent/(condition+'_MANIFEST.json')).exists():continue
            subprocess.run(['/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python','-m','tools.contact_coordination.target_dataset',
                '--manifest',str(out/'matched_datasets/MEMBERSHIP.json'),'--condition',condition,'--destination',str(destination)],check=True,cwd=ROOT,
                env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false'))
        return dict(status='MATCHED_G1_DATASETS_CONSTRUCTED',counts=counts)
    if stage=='effective_data_audit' and (out/'matched_datasets/A_MANIFEST.json').exists() and (out/'matched_datasets/B_MANIFEST.json').exists():
        import subprocess
        from .io import ROOT
        subprocess.run(['/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python','-m','tools.contact_coordination.target_dataset_audit','--run-dir',str(out),
            '--manifest-a',str(out/'matched_datasets/A_MANIFEST.json'),'--manifest-b',str(out/'matched_datasets/B_MANIFEST.json')],check=True,cwd=ROOT)
        return read(out/'effective_data_audit/TARGET_DATASET_PARITY.json')
    if stage=='ACT_train_or_verified_reuse' and (out/'effective_data_audit/TARGET_DATASET_PARITY.json').exists():
        from .act_training import prepare,train
        prepare(out,out/'matched_datasets/A_MANIFEST.json',out/'matched_datasets/B_MANIFEST.json',out/'effective_data_audit/TARGET_DATASET_PARITY.json')
        return train(out,resume)
    if stage in ('ACT_interface_test','final_freeze','ACT_A35_B35') and (out/'act_training/ACT_TRAINING_COMPLETE.json').exists():
        from .policy_study import sanity,freeze,run
        return {'ACT_interface_test':sanity,'final_freeze':freeze,'ACT_A35_B35':run}[stage](out)
    if stage=='reference_coupling10':
        from .reference_study import run
        return run(out,resume)
    if stage=='analysis':
        from .paper_results import analyze
        from .mechanistic_metrics import run
        run(out)
        return analyze(out)
    if stage=='figures':
        from .paper_figures import generate
        return generate(out)
    if stage=='replays':
        from .paper_replays import run
        return run(out)
    if stage=='report':
        from .paper_handoff import write_report
        return write_report(out)
    if stage=='final_verification':
        from .paper_handoff import verify
        return verify(out)
    return dict(status='UPSTREAM_PREREQUISITE_PENDING', stage=stage,
                reasons=['Read current generation/supervision/checkpoint evidence below; no downstream outcomes are fabricated.'],
                training_split_authorization_resolved=True,
                automatic_downstream_outcomes_created=False,
                current_progress=status(out))
