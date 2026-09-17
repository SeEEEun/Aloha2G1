"""One resumable method/source conversion over the existing local runners."""
from pathlib import Path
import fcntl
import time
from .io import read,record,atomic_json


def attempt(out,source_id,condition,scope='TRAIN_pilot',execute=True,resume=False):
    from .generalization_gate import require
    require(out, scope)
    if scope in ('TRAIN40_conversion','REFERENCE_coupling10'):
        from .generation_study import verify_freeze
        verify_freeze(out)
    if condition in ('B','B_NO_COUPLING'):
        from .interaction_chain import attempt as repaired_attempt
        method='C_COUPLED' if condition=='B' else 'B_INDEPENDENT'
        row=repaired_attempt(out,source_id,method,scope)
        row.update(condition=condition,scope=scope)
        if not execute or not row['full_task_plan']:return row
        if scope=='ARCHITECTURE_REPAIR':raise ValueError('Architecture physics runs only after structural stage receipts')
        from .full_attempt import prepare as prepare_recording,launch as execute_recording
        from .abc_score import score as score_recording
        plan=Path(row['plan']);name=source_id+'_'+method+'_'+record(plan/'COMMANDS.npz')['sha256'][:12]
        folder=out/'physical_attempts'/name
        if not folder.exists():folder=prepare_recording(out,plan,name,method)
        with (folder/'.execution.lock').open('a+') as execution_lock:
            fcntl.flock(execution_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            process=execute_recording(folder,resume=True)
            if process['returncode']!=0:
                row.update(terminal='INFRASTRUCTURE_INVALID',first_failure='PHYSICS_PROCESS')
            else:
                measured=score_recording(folder,Path(row['context']))
                row.update(terminal=measured['terminal'],physical_run=True,attempt=str(folder),
                    task_success=measured['stages']['FULL_TASK'],stages=measured['stages'],physical_validity=measured['physical_validity'],first_failure=measured['first_failed_stage'])
            atomic_json(folder/'CONVERSION_RESULT.json',row)
        return row
    from .episode_context import prepare
    context=prepare(out,source_id,condition,scope)
    path=context/'INSTANCE_RESULT.json'
    with (context/'.instance.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            row=read(path)
            if not resume:raise FileExistsError(path)
            for item in row.get('artifacts',[]):
                if record(item['path'])!=item:raise ValueError('Changed saved instance artifact')
            if row['terminal']!='PLAN_VALID_PENDING_PHYSICS' or not execute:return row
        else:
            started=time.monotonic()
            row=dict(source_id=source_id,condition=condition,context=str(context),scope=scope,
                     full_task_plan=False,physical_run=False,task_success=None,artifacts=[])
            acquisition=context/'prototype'/source_id/'morphology_acquisition_v4/RESULT.json'
            if condition=='A':
                from .wrist_reference import acquisition as build_acquisition
            else:
                from .acquisition_plan import build as build_acquisition
            build_acquisition(context,context/'target_repair/CONTACT_CALIBRATION.json')
            acq=read(acquisition);row['artifacts'].append(record(acquisition));row['acquisition_status']=acq['status']
            if acq['status']!='ACQUISITION_PLAN_BUILT_PENDING_FINAL_VALIDATION':
                row.update(terminal='NO_PLAN_WITHIN_FIXED_BUDGET' if scope=='TRAIN40_conversion' else 'TRAIN_NO_PLAN_WITHIN_BOUNDED_POLICY',first_failure='ACQUISITION_CONNECTION',planning_seconds=time.monotonic()-started)
                atomic_json(path,row);return row
            carry=context/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json'
            loaded=context/'target_repair/LOADED_CARRY_GEOMETRY_f83d2e29156f.json'
            if condition=='A':
                from .wrist_reference import full_task
                connection=full_task(context,carry,loaded)
            else:
                from .handoff_repair import fit_region
                from .full_task_plan import build
                fit_region(context,'translation',carry,'calibrated_gravity','receiver_acquisition_intent')
                connection=build(context,giver_release_policy='middle_first',receiver_departure=True,loaded_geometry=loaded,enable_coupling=condition!='B_NO_COUPLING')
            result=read(connection/'RESULT.json');row.update(connection=str(connection),planning_seconds=time.monotonic()-started)
            row['artifacts'].append(record(connection/'RESULT.json'))
            if result['status']!='FULL_PATH_BUILT':
                row.update(terminal='NO_PLAN_WITHIN_FIXED_BUDGET' if scope=='TRAIN40_conversion' else 'TRAIN_NO_PLAN_WITHIN_BOUNDED_POLICY',first_failure='FULL_TASK_CONNECTION')
                atomic_json(path,row);return row
            from .physical_attempt import export_full,validate_full
            plan=connection/'physical_plan'
            if not plan.exists():plan=export_full(context,connection)
            validation=read(plan/'FINAL_COMMAND_VALIDATION.json') if (plan/'FINAL_COMMAND_VALIDATION.json').exists() else validate_full(context,plan)
            row.update(plan=str(plan),full_task_plan=validation['status']=='VALID')
            row['artifacts'] += [record(plan/name) for name in ('PLAN.json','COMMANDS.npz','FINAL_COMMAND_VALIDATION.json','SOURCE_SCENE.json')]
            row.update(terminal='PLAN_VALID_PENDING_PHYSICS' if row['full_task_plan'] else 'NO_PLAN_POST_RETIME_VALIDATION',first_failure=None if row['full_task_plan'] else 'POST_RETIME_GEOMETRY')
            atomic_json(path,row)
        if execute and row['terminal']=='PLAN_VALID_PENDING_PHYSICS':
            # A receives the same complete-horizon controller/PhysX/scorer as
            # B/C; source wrist representation is its only spatial distinction.
            from .full_attempt import prepare as prepare_recording,launch as execute_recording
            from .abc_score import score as score_recording
            plan=Path(row['plan']);name=source_id+'_A_WRIST_'+record(plan/'COMMANDS.npz')['sha256'][:12]
            folder=out/'physical_attempts'/name
            if not folder.exists():folder=prepare_recording(out,plan,name,'A_WRIST')
            process=execute_recording(folder,resume=True)
            row.update(physical_run=(folder/'event_log.npz').exists(),attempt=str(folder));row['artifacts'].append(record(folder/'PROCESS.json'))
            if process['returncode']!=0 or not row['physical_run']:
                row.update(terminal='INFRASTRUCTURE_INVALID',first_failure='PHYSICS_PROCESS')
            else:
                outcome=score_recording(folder,context)
                row.update(terminal=outcome['terminal'],task_success=outcome['stages']['FULL_TASK'],physical_validity=outcome['physical_validity'],stages=outcome['stages'],first_failure=outcome['first_failed_stage'])
                row['score']=record(folder/'ABC_NOMINAL_SCORE.json');row['artifacts'].append(row['score'])
            atomic_json(path,row)
        return row


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--source-id',required=True)
    p.add_argument('--condition',choices=['A','B','B_NO_COUPLING'],required=True);p.add_argument('--scope',choices=['TRAIN_pilot','TRAIN40_conversion','REFERENCE_coupling10'],default='TRAIN_pilot');p.add_argument('--plan-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(attempt(a.run_dir,a.source_id,a.condition,a.scope,not a.plan_only,a.resume))
