"""Serial actual PhysX verification after the planning improvement gate."""
from pathlib import Path
import numpy as np
from .io import read,record,atomic_json
from .practical_study import assert_backend,now


def execute_case(out,area,plan,stage):
    """Reuse the repaired complete-horizon interface and unchanged task scorer."""
    from .architecture_physics import run as execute
    from .full_attempt import prepare
    from .scientific_cache import digest
    from .source_guided_execution_watchdog import preserve_incomplete,configure
    from .incidental_contact import policy
    sid=plan['source_id'];method=plan['method_key'];area=Path(area)
    report_name='physical_case_reports/'+sid+'_'+method+'.json'
    report=area/report_name
    if not plan['full_task_plan']:
        value=dict(source_id=sid,method_key=method,complete_plan=False,physics_executed=False,
            status='NO_COMPLETE_COMMAND',first_failure=plan['first_failure'],context=plan['context'],plan=None)
        atomic_json(report,dict(status='PASS',rows=[value]));return value
    command=record(Path(plan['plan'])/'COMMANDS.npz')
    code=[record(Path(__file__).parent/name) for name in ('full_attempt.py','full_attempt_runtime.py','full_attempt_physics.py',
        'phase_clock_runtime.py','contact_command_retiming.py','phase_physics.py','execution_timing.py','physics_capture.py','pregrasp_protection.py')]
    token=digest(dict(command=command,code=code))[:12];name=sid+'_'+method+'_'+token
    folder=area/'physical_attempts'/name
    recovery=preserve_incomplete(folder,area)
    if not folder.exists():
        folder=prepare(area,Path(plan['plan']),name,method)
        policy_file=folder/'input/INCIDENTAL_CONTACT_POLICY.json';atomic_json(policy_file,policy(Path(plan['context'])))
        deps=read(folder/'DEPENDENCIES.json');deps['files']+=[record(policy_file),record(Path(__file__).parent/'incidental_contact.py'),record(Path(__file__).parent/'practical_parameters.py')]
        atomic_json(folder/'DEPENDENCIES.json',deps)
    if recovery:atomic_json(folder/'INFRASTRUCTURE_RECOVERY_LINK.json',dict(archive=record(recovery)))
    configure(folder)
    if not (area/'GOLDEN_PLANNING.json').exists():atomic_json(area/'GOLDEN_PLANNING.json',dict(rows=[]))
    atomic_json(out/'WORK_PROGRESS.json',dict(stage=stage,source_id=sid,method=method,recipe=area.name,
        substage='Actual PhysX full command recording',engine_log=str(folder/'engine.log')))
    # Completed failed tasks are valid cached executions; no outcome-based retry.
    try:
        execute(area,require_complete_coverage=False,collect_structure=False,plans_override=[plan],report_name=report_name)
    except RuntimeError:
        abort=folder/'NUMERICAL_ABORT.json'
        if not abort.exists() or not (folder/'event_log.npz').exists():raise
        # A genuine measured abort is an official failure, not a no-plan case.
        value=dict(source_id=sid,method_key=method,complete_plan=True,physics_executed=True,
            status='NUMERICAL_ABORT',folder=str(folder),first_failure='NUMERICAL_ABORT',abort=record(abort),
            context=plan['context'],plan=plan['plan'],physical_validity=False,trace=record(folder/'event_log.npz'),
            stages={k:False for k in ('GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','TRANSPORT','BIN_ENTRY','BIN_SETTLE','FULL_TASK')},
            command_horizon_verified=False,genuine_abort_explains_shorter_horizon=True)
        atomic_json(report,dict(status='PASS',rows=[value]))
    value=read(report)['rows'][0]
    if value.get('status')!='NUMERICAL_ABORT':
        assert value['command_horizon_verified'] and value['post_failure_arm_sequence_preserved']
    value['incidental_contact_policy']=record(folder/'input/INCIDENTAL_CONTACT_POLICY.json')
    atomic_json(report,dict(status='PASS',rows=[value],architecture_frozen=True));return value


def run(out):
    assert_backend(out);rank=read(out/'RECIPE_RANKING.json');assert rank['planning_improved']
    study=read(out/'PRACTICAL_STUDY.json');sources=study['physical_source_ids'];plans=read(out/'PLANNING_SWEEP.json')['rows']
    selected=rank['top_recipes'];assert 1<=len(selected)<=2
    scheduled=[(recipe,sid,m) for recipe in selected for sid in sources for m in study['methods']]
    lookup={(r['recipe'],r['source_id'],r['method']):r['plan'] for r in plans};rows=[];artifacts=[]
    path=out/'TOP_RECIPE_PHYSICS.json'
    for index,(recipe,sid,method) in enumerate(scheduled):
        area=out/'recipes'/recipe
        row=execute_case(out,area,lookup[(recipe,sid,method)],'top_recipe_physics');row=dict(row,recipe=recipe)
        rows.append(row);artifacts.append(area/'physical_case_reports'/(sid+'_'+method+'.json'))
        atomic_json(path,dict(status='IN_PROGRESS',rows=rows,completed=index+1,scheduled=len(scheduled)))
        atomic_json(out/'WORK_PROGRESS.json',dict(stage='top_recipe_physics',recipe=recipe,source_id=sid,method=method,
            substage='Completed predeclared physical case',completed_work=index+1,remaining_work=len(scheduled)-index-1))
    atomic_json(path,dict(status='PASS',rows=rows,scheduled=len(scheduled),physical_source_ids=sources,
        recipe_selection_used_physics=False,subset_selected_before_new_outcomes=True,completed_at=now()))
    return [path,*artifacts]
