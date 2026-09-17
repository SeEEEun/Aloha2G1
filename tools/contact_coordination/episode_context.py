"""Small immutable input contexts for the existing per-source converter runners.

Only manifests/calibration are staged here. No trajectory, scene pose or outcome
is copied from the successful prototype to another source.
"""
from pathlib import Path
import fcntl
import shutil
from .io import read,record,atomic_json,fingerprint


def prepare(out,source_id,condition,scope='TRAIN_pilot'):
    from .generalization_gate import require
    require(out, scope)
    if condition not in ('A','B','B_NO_COUPLING'):raise ValueError('Unknown representation')
    if scope not in ('TRAIN_pilot','TRAIN40_conversion','REFERENCE_coupling10','SOURCE_SENSITIVITY','TRAIN_calibration','TRAIN40_ABC','DEV35_ABC_reference','ARCHITECTURE_REPAIR'):raise ValueError('Unknown execution scope')
    split=read(out/'SPLIT_CONTRACT.json');selection=read(out/'bootstrap/SELECTION.json')
    allowed=split['evaluation_source_ids'] if scope=='DEV35_ABC_reference' else split['ablation_source_ids'] if scope=='REFERENCE_coupling10' else split['authorized_training_source_ids']
    if split['status']!='AUTHORIZED_TRAIN40_DEV35' or source_id not in allowed:
        raise ValueError('Not an authorized source for this scope')
    if scope not in ('REFERENCE_coupling10','DEV35_ABC_reference') and source_id in split['evaluation_source_ids']:raise ValueError('TRAIN/DEV overlap')
    if scope=='TRAIN_pilot' and source_id not in [selection['prototype_source_id'],*selection['additional_train_source_ids']]:
        raise ValueError('Recording is outside the predeclared TRAIN pilot')
    source=out/'source_phase'/source_id
    phase=read(source/'PHASE_RECORD.json')
    if phase['source_id']!=source_id:raise ValueError('Source identity mismatch')
    if phase['hand_roles']!={'giver':'left','receiver':'right'}:
        raise ValueError('Source roles do not describe the declared left-to-right doll task')
    common=['target_repair/CONTACT_CALIBRATION.json','target_repair/CONTACT_SEPARATION_CANDIDATES.json',
            'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json','target_repair/LOADED_CARRY_GEOMETRY_f83d2e29156f.json',
            'target_repair/CALIBRATED_HANDOFF_GRAVITY.json','target_repair/CALIBRATED_RIGHT_CARRY_GRAVITY.json',
            'target_repair/runtime_bin150/RUNTIME_HULLS.json','target_repair/runtime_bin150/RUNTIME_HULL_VERTICES.npz']
    source_files=[p for p in source.iterdir() if p.is_file() and p.suffix in ('.json','.npz')]
    if (out/'target_repair/TASK_FRAME_NORMALIZATION.json').exists():
        common.append('target_repair/TASK_FRAME_NORMALIZATION.json')
    if (out/'CALIBRATION_PARAMETERS.json').exists():
        common.append('CALIBRATION_PARAMETERS.json')
    if (out/'PRACTICAL_PARAMETERS.json').exists():common.append('PRACTICAL_PARAMETERS.json')
    copied=[out/relative for relative in common]+source_files
    implementation_dir=Path(__file__).parent
    implementations=[implementation_dir/name for name in ('episode_context.py','acquisition_plan.py','handoff_repair.py',
        'full_task_plan.py','planner.py','runtime_hulls.py','loaded_contact_geometry.py','physical_attempt.py',
        'phase_clock_runtime.py','wrist_reference.py')]
    inputs=copied+[out/'SPLIT_CONTRACT.json',out/'bootstrap/SELECTION.json',*implementations]
    from .scientific_cache import key as cache_key
    signature=cache_key(out,source_id,'episode_context',dict(condition=condition,scope=scope))
    dependencies=[record(p) for p in inputs]
    destination=out/scope/'contexts'/source_id/condition/signature[:12]
    destination.mkdir(parents=True,exist_ok=True)
    contract=dict(source_id=source_id,condition=condition,scope=scope,scientific_cache_key=signature,dependencies=dependencies,
        source_scene=phase['initial_object_pose_world'],scene_independent_of_method=True,
        source_roles=phase['hand_roles'],trajectory_copied=False,implementation=record(__file__))
    with (destination/'.input_context.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=destination/'INPUT_CONTEXT.json'
        if receipt.exists():
            saved=read(receipt)
            if saved['scientific_cache_key']!=signature:raise ValueError('Changed immutable scientific context')
            staged_key=cache_key(destination,source_id,'episode_context',dict(condition=condition,scope=scope))
            if staged_key!=signature:raise ValueError('Changed staged scientific source/calibration input')
            return destination
        for path in copied:
            target=destination/path.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        staged=dict(selection,prototype_source_id=source_id,context_scope=scope,
                    parent_selection=record(out/'bootstrap/SELECTION.json'),representation_condition=condition)
        atomic_json(destination/'bootstrap/SELECTION.json',staged)
        atomic_json(destination/'SPLIT_CONTRACT.json',split)
        atomic_json(receipt,contract)
    return destination


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--source-id',required=True)
    p.add_argument('--condition',choices=['A','B','B_NO_COUPLING'],required=True)
    p.add_argument('--scope',choices=['TRAIN_pilot','TRAIN40_conversion'],default='TRAIN_pilot')
    a=p.parse_args();print(prepare(a.run_dir.resolve(),a.source_id,a.condition,a.scope))
