"""Thin local stage entry point with explicit unresolved downstream gates.

This is a resumable prerequisite implementation, not a completed ACT study
or a service. It never fabricates datasets, schedules or policy outcomes.
"""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .io import ROOT,read,record,atomic_json,atomic_text

STAGES=['inspect_splits','control','target_repair','prototype','TRAIN_pilot',
    'generation_freeze','matched_dataset_realization','effective_data_audit',
    'ACT_train_or_verified_reuse','ACT_interface_test','final_freeze','ACT_A35_B35',
    'reference_coupling10','analysis','figures','replays','report','final_verification','status','all']


def verify(out):
    from .act_interface_verify import run as verify_interface
    interface=verify_interface(out)
    n=read(out/'NUMERIC_SUMMARY.json')
    assert n['ACT_A_main_trials']==n['ACT_B_main_trials']==0
    assert n['source_conditioned_full_task']=='NOT_DEMONSTRATED'
    import csv,numpy as np
    with (out/'ACT_DEV35_PER_EPISODE_RESULTS.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==70 and all(r['terminal']=='NOT_ATTEMPTED_UPSTREAM' and r['physically_run']=='0' for r in rows)
    sid=n['source_id'];folder=out/'prototype'/sid/'morphology_acquisition_v4/physics_attempt_01'
    score=read(folder/'INDEPENDENT_ACQUISITION_SCORE.json');trace=dict(np.load(folder/'event_log.npz'))
    assert score['artifact_sha256']['event_log']==record(folder/'event_log.npz')['sha256']
    assert len(trace['control_frame'])==3488 and score['maximum_com_lift_mm']>=50 and score['natural_release']
    patch=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
    preserved=patch==(out/'bootstrap/tracked_changes_initial.patch').read_bytes()
    assert preserved,'Pre-existing tracked edits changed'
    videos=read(out/'replays/REPLAY_STATUS.json')
    assert len(videos['actual_short_replays'])==3
    for video in videos['actual_short_replays']:
        assert record(video['video']['path'])==video['video']
    required=['STUDY_CONTRACT.md','SPLIT_AUDIT.csv','SPLIT_CONTRACT.json','REUSE_MAP.md',
        'EFFECTIVE_SUPERVISION_DIFF.md','DATASET_PARITY.md','METHOD_PARITY.md','FIRST_FAILURE_CHAIN.md',
        'TABLE_CONVERSION_AND_DATASET_PROVENANCE.csv','TABLE_ACT_A_VS_ACT_B_DEV35.csv',
        'TABLE_REFERENCE_COUPLING_ABLATION_PAIRED10.csv','ACT_DEV35_PER_EPISODE_RESULTS.csv',
        'FINAL_REPORT.md','METHODS_DRAFT_KO.md','RESULTS_AND_LIMITATIONS_KO.md',
        'PAPER_FIGURE_DIRECTION.md','REPRODUCE.md','REGRESSION_TESTS.log']
    files=[record(out/f) for f in required]
    result=dict(status='PARTIAL_PACKAGE_EVIDENCE_VERIFIED',terminal=n['terminal'],
        prior_tracked_edits_preserved=preserved,primary_ACT_comparison_completed=False,
        source_full_task_demonstrated=False,compatible_training_datasets=0,
        verified_legacy_interface_diagnostics=len(interface),actual_short_replays=3,
        partial_report_files=files,missing_primary_videos=videos['required_primary_videos'],
        missing_reference_ablation_clip=videos['reference_ablation_clip'],
        completion_inferred_from_file_existence=False)
    atomic_json(out/'FINAL_VERIFICATION.json',result);return result


def dispatch(out,stage,resume):
    from .generalization_gate import require
    require(out, stage)
    if (out/'TARGETED_REPAIR_CONTRACT.json').exists():
        from .targeted_progress import dispatch as targeted
        return targeted(out,stage,resume)
    # Continue the explicitly authorized TRAIN40 study without resurrecting
    # historical TRAIN50 gates or the old prototype-only report assumptions.
    if read(out/'SPLIT_CONTRACT.json').get('status')=='AUTHORIZED_TRAIN40_DEV35':
        from .current_progress import dispatch as current_dispatch
        return current_dispatch(out,stage,resume)
    if stage=='inspect_splits':
        if resume and (out/'SPLIT_CONTRACT.json').exists():
            d=read(out/'SPLIT_CONTRACT.json');assert record(d['manifest']['path'])==d['manifest'];return d
        from .act_split_audit import audit
        return audit(out)
    if stage=='control':
        from .act_control_capture import run
        return run(out,resume)
    if stage=='target_repair':
        if resume and (out/'target_repair/TARGET_REPAIR_STATE.json').exists():
            d=read(out/'target_repair/TARGET_REPAIR_STATE.json')
            for dep in d['dependencies']:assert record(dep['path'])==dep
            return d
        raise RuntimeError('Existing bounded target repair is preserved. Use --resume to inspect its exact version; further development requires a new recorded code/config version, not an unchanged search restart.')
    if stage=='effective_data_audit':
        if resume and (out/'effective_data_audit/EFFECTIVE_SUPERVISION_AUDIT.json').exists():
            d=read(out/'effective_data_audit/EFFECTIVE_SUPERVISION_AUDIT.json')
            for method in d['methods'].values():
                for dep in method['checkpoint_files']:assert record(dep['path'])==dep
            return dict(status='INCOMPATIBLE_LEGACY_PAIR_RECONFIRMED')
        from .act_effective_audit import run
        return run(out)
    if stage=='ACT_interface_test':
        from .act_interface_probe import run
        from .act_interface_verify import run as audit
        run(out,resume=resume);return audit(out)
    if stage=='analysis':
        from .act_prerequisite_report import analysis
        return analysis(out)
    if stage=='report':
        from .act_prerequisite_report import report
        return report(out)
    if stage=='figures':
        from .act_prerequisite_media import figures,closeup,direction
        r=figures(out);closeup(out);direction(out);return r
    if stage=='replays':
        from .act_prerequisite_media import replays
        return replays(out,resume)
    if stage=='final_verification':return verify(out)
    if stage=='status':return read(out/'RUN_STATE.json')
    # The source prototype gate cannot be satisfied by a reference card or by
    # the separate legacy checkpoint diagnostic. No downstream ready claim.
    return dict(status='NOT_ATTEMPTED_UPSTREAM',stage=stage,
        reasons=['Source-conditioned full task not demonstrated: PLACE remains invalid',
                 'Requested disjoint50-source training split lacks authorization/data',
                 'Matched dynamic demonstrations and compatible selected ACT pair absent'],
        downstream_implementation_status='NOT_YET_IMPLEMENTED_OR_QUALIFIED_BEYOND_FAIL_CLOSED_GATE',
        automatic_training_or_evaluation_launched=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    out=args.run_dir.resolve();out.relative_to(ROOT/'outputs/contact_coordination_hybrid_act')
    assert (out/'STUDY_CONTRACT.md').exists(),'Bootstrap the actual existing run before using stages'
    with (out/'.hybrid_act.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=[s for s in STAGES if s not in ('status','all')] if args.stage=='all' else [args.stage]
        for stage in stages:
            result=dispatch(out,stage,args.resume)
            atomic_json(out/'stage_receipts'/f'{stage}.json',dict(stage=stage,result=result,entrypoint=record(__file__),time=time.time()))
            with (out/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),stage=stage,status=result.get('status') if isinstance(result,dict) else 'RECORDED'))+'\n')
            print(stage, result.get('status',result.get('terminal','RECORDED')) if isinstance(result,dict) else 'RECORDED',flush=True)
            if isinstance(result,dict) and result.get('continue_downstream') is False:break


if __name__=='__main__':main()
