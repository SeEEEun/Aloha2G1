"""Execute only complete repaired plans; retain failures to the true horizon."""
from pathlib import Path
import numpy as np
from .io import read,record,atomic_json


def run(out,require_complete_coverage=True,collect_structure=True,plans_override=None,report_name='PHYSICAL_VERIFICATION.json',progress_offset=0,total_scheduled=None):
    import unittest
    from .tests.test_architecture_physics_cache import PhysicsResumeTests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PhysicsResumeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise AssertionError('Physics resume dependency regression failed')
    resume_evidence=out/'PHYSICS_RESUME_DEPENDENCY_REGRESSION.json'
    atomic_json(resume_evidence,dict(status='PASS',test_count=result.testsRun,
        test=record(Path(__file__).parent/'tests/test_architecture_physics_cache.py'),
        implementation=record(Path(__file__).parent/'full_attempt.py')))
    from .architecture_evidence import collect
    structural_artifacts=collect(out) if collect_structure else []
    from .full_attempt import prepare,launch
    from .abc_score import score
    golden=read(out/'GOLDEN_PLANNING.json')['rows'];coverage=read(out/'COVERAGE8_PLANNING.json')['rows'] if plans_override is None else []
    rows=[]
    # All Coverage8 complete commands are executed so each available tile has
    # a measured full attempt; no outcomes select the physical subset.
    scheduled=[*golden,*coverage] if plans_override is None else plans_override
    for index,plan in enumerate(scheduled):
        sid=plan['source_id'];method=plan['method_key']
        row=dict(source_id=sid,method_key=method,plan=plan.get('plan'),context=plan['context'],
            complete_plan=plan['full_task_plan'],physics_executed=False,eligible_for_final_paper_dataset=False)
        if not plan['full_task_plan']:
            row.update(status='NO_COMPLETE_COMMAND',first_failure=plan['first_failure']);rows.append(row);continue
        command=record(Path(plan['plan'])/'COMMANDS.npz')
        from .scientific_cache import digest
        code=[record(Path(__file__).parent/name) for name in ('full_attempt.py','full_attempt_runtime.py','full_attempt_physics.py',
            'phase_clock_runtime.py','contact_command_retiming.py','phase_physics.py','execution_timing.py','physics_capture.py','pregrasp_protection.py')]
        token=digest(dict(command=command,code=code))[:12]
        name=sid+'_'+method+'_'+token;folder=out/'physical_attempts'/name
        recovery=None;watchdog=None
        if (out/'SOURCE_GUIDED_RRT_REBUILD.json').exists():
            from .source_guided_execution_watchdog import preserve_incomplete,configure
            recovery=preserve_incomplete(folder,out)
        equivalence=None
        if not folder.exists():
            from .source_guided_physics_cache import find_equivalent
            reused=find_equivalent(out,Path(plan['plan']),method) if (out/'SOURCE_GUIDED_RRT_REBUILD.json').exists() else None
            if reused:folder,equivalence=reused
            else:folder=prepare(out,Path(plan['plan']),name,method)
        recovery_link=folder/'INFRASTRUCTURE_RECOVERY_LINK.json'
        if recovery:atomic_json(recovery_link,dict(archive=record(recovery)))
        elif recovery_link.exists():
            evidence=read(recovery_link)['archive'];assert record(evidence['path'])==evidence
            recovery=Path(evidence['path'])
        if (out/'SOURCE_GUIDED_RRT_REBUILD.json').exists():watchdog=configure(folder)
        atomic_json(out/'WORK_PROGRESS.json',dict(stage='small_physics_validation',source_id=sid,method=method,
            substage='PhysX complete command horizon',completed_work=progress_offset+index,
            remaining_work=(total_scheduled or len(scheduled))-progress_offset-index,engine_log=str(folder/'engine.log')))
        result=launch(folder,resume=True)
        if result['returncode']!=0 or not (folder/'event_log.npz').exists():
            row.update(status='INFRASTRUCTURE_FAILURE',process=result,folder=str(folder));rows.append(row)
            atomic_json(out/report_name,dict(status='FAIL',rows=rows))
            raise RuntimeError('PhysX process failed: '+str(folder/'engine.log'))
        recording=read(folder/'FULL_ATTEMPT_RECORDING.json');trace=np.load(folder/'event_log.npz')
        expected=recording['requested_recording_frames'];actual=int(trace['control_frame'][-1])+1
        if actual!=expected or recording['recording_end_reason']!='COMPLETE_DECLARED_HORIZON':
            row.update(status='NUMERICAL_ABORT',recorded=actual,expected=expected,folder=str(folder));rows.append(row)
            atomic_json(out/report_name,dict(status='FAIL',rows=rows))
            raise RuntimeError('Physical recording did not reach true command horizon: '+str(folder))
        if recording['official_admission_stop'] is not None:raise AssertionError('Task failure froze official commands')
        from .contact_command_retiming import audit_issued_commands
        timing=audit_issued_commands(trace)
        atomic_json(folder/'ISSUED_COMMAND_TIMING_AUDIT.json',timing)
        if timing['status']!='PASS':raise AssertionError('RETIMING_FAIL: issued contact/arm command limits: '+str(folder))
        if (folder/'ROBOT_OBJECT_CONTACTS.json').exists():
            from .pregrasp_protection import audit_trial
            audit_trial(folder,trace)
        outcome=score(folder,Path(plan['context']))
        from .contact_command_geometry import audit as audit_contact_geometry
        contact_geometry=audit_contact_geometry(folder,Path(plan['context']),trace)
        with np.load(Path(plan['plan'])/'COMMANDS.npz') as commands:
            # Exact issued arm setpoints prove no admission hold replaced the
            # remainder of the pre-existing motion after a missed/dropped object.
            indices=np.r_[np.flatnonzero(np.diff(trace['control_frame'])!=0),len(trace['control_frame'])-1]
            nominal=len(commands['commanded_q_rad'])
            np.testing.assert_allclose(trace['EXECUTED_COMMAND'][indices[:nominal],:14],commands['commanded_q_rad'][:,:14],atol=1e-6,rtol=0)
            failure=recording['first_failure_latched'];motion=None
            if failure:
                begin=int(failure['control_frame']);remaining=indices[begin:nominal]
                if len(remaining)>1:
                    expected_motion=float(np.max(np.ptp(commands['commanded_q_rad'][begin:,:14],axis=0)))
                    measured_motion=float(np.max(np.ptp(trace['MEASURED_Q'][remaining,:14],axis=0)))
                    motion=dict(commanded_arm_range_rad=expected_motion,measured_arm_range_rad=measured_motion)
                    if expected_motion>.01 and measured_motion<.001:raise AssertionError('Physical arm trace froze despite remaining arm commands')
        first=outcome.get('first_failed_stage')
        causal={'GRASP':'PHYSICS_GRASP_FAIL','LIFT':'PHYSICS_GRASP_FAIL','HANDOFF':'PHYSICS_HANDOFF_FAIL',
            'RIGHT_OWNERSHIP':'PHYSICS_OWNERSHIP_FAIL','TRANSPORT':'PHYSICS_OWNERSHIP_FAIL',
            'BIN_ENTRY':'PHYSICS_PLACE_FAIL','BIN_SETTLE':'PHYSICS_PLACE_FAIL'}.get(first)
        from .physical_failure_evidence import failure_events
        failures=failure_events(folder,recording,outcome,expected,trace)
        if not outcome['stages']['FULL_TASK'] and not failures:raise AssertionError('Failed physical trial lacks causal evidence')
        row.update(status='EXECUTED_COMPLETE_HORIZON',physics_executed=True,folder=str(folder),
            infrastructure_recovery=record(recovery) if recovery else None,
            execution_wall_watchdog=record(watchdog) if watchdog else (record(folder/'EXECUTION_WALL_WATCHDOG.json') if (folder/'EXECUTION_WALL_WATCHDOG.json').exists() else None),
            execution_equivalence=record(equivalence) if equivalence else None,
            stages=outcome['stages'],physical_validity=outcome['physical_validity'],
            first_failure=failures[0]['causal_label'] if failures else None,
            first_failure_detail=failures[0] if failures else None,physical_failure_events=failures,first_task_failure=causal,
            recording=record(folder/'FULL_ATTEMPT_RECORDING.json'),trace=record(folder/'event_log.npz'),
            issued_command_timing=record(folder/'ISSUED_COMMAND_TIMING_AUDIT.json'),
            adaptive_command_geometry=record(contact_geometry),
            score=record(folder/'ABC_NOMINAL_SCORE.json'),command_horizon_verified=True,
            post_failure_arm_sequence_preserved=True,post_failure_measured_motion=motion,selected_candidate_id=plan['selected_candidate_id'])
        rows.append(row);atomic_json(out/report_name,dict(status='IN_PROGRESS',rows=rows))
    fixed=read(out/'COVERAGE8.json')
    for method in ('B_INDEPENDENT','C_COUPLED') if require_complete_coverage else ():
        selected=[r for r in rows if r['method_key']==method and r['physics_executed']]
        if not any(r['source_id']==fixed['golden'] for r in selected):raise AssertionError('Golden physics absent')
        if len({r['source_id'] for r in selected if r['source_id']!=fixed['golden']})<3:raise AssertionError('Non-Golden physical coverage incomplete')
    path=out/report_name
    atomic_json(path,dict(status='PASS',task_success_requirement='Architecture execution, not100% task success',rows=rows,
        no_episode_tuning=True,all_complete_commands_executed=True))
    return [path,resume_evidence,*structural_artifacts,*[Path(r['recording']['path']) for r in rows if r['physics_executed']],
            *[Path(r['issued_command_timing']['path']) for r in rows if r['physics_executed']],
            *[Path(r['adaptive_command_geometry']['path']) for r in rows if r['physics_executed']],
            *[Path(r['score']['path']) for r in rows if r['physics_executed']],
            *[Path(r['execution_equivalence']['path']) for r in rows if r.get('execution_equivalence')],
            *[Path(r['infrastructure_recovery']['path']) for r in rows if r.get('infrastructure_recovery')],
            *[Path(r['execution_wall_watchdog']['path']) for r in rows if r.get('execution_wall_watchdog')]]
