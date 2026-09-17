"""Process-wall-time recovery; never changes simulated time or commands."""
from pathlib import Path
import time
import numpy as np
from .io import read,record,atomic_json


def preserve_incomplete(folder,out):
    folder=Path(folder);out=Path(out)
    if not folder.exists():return None
    process=read(folder/'PROCESS.json') if (folder/'PROCESS.json').exists() else None
    # A completed simulation, including a failed task, is never repeated here.
    if process and process['returncode']==0:return None
    manifest=[dict(relative_path=str(p.relative_to(folder)),**record(p)) for p in sorted(folder.rglob('*')) if p.is_file()]
    archive=out/'INFRASTRUCTURE_INTERRUPTED_ATTEMPTS'/(folder.name+'_'+str(time.time_ns()))
    archive.parent.mkdir(parents=True,exist_ok=True);folder.rename(archive)
    receipt=archive/'INFRASTRUCTURE_ARCHIVE.json'
    atomic_json(receipt,dict(original_folder=str(folder),archive=str(archive),process=process,
        reason='FAILED_OR_INTERRUPTED_PROCESS_WITHOUT_COMPLETE_TRIAL',files_before_move=manifest,
        official_physical_task_result=False,task_outcomes_used_for_retry=False,
        raw_evidence_preserved=True,planner_and_commands_unchanged=True))
    return receipt


def configure(folder):
    folder=Path(folder)
    if (folder/'PROCESS.json').exists():return None
    path=folder/'INVOCATION.json';invocation=read(path)
    snapshot=folder/'DEFAULT_PROCESS_INVOCATION.json'
    if not snapshot.exists():atomic_json(snapshot,invocation)
    before=record(snapshot)
    with np.load(folder/'input/COMMANDS.npz') as data:frames=len(data['stage'])
    # Fixed common infrastructure allowance: two wall seconds per command
    # frame plus 120 seconds startup, with the previous 1200-second floor.
    # This is not a planning budget or a physical/control timing parameter.
    timeout=max(1200,120+2*frames)
    prior=invocation['timeout_s'];invocation['timeout_s']=timeout
    atomic_json(path,invocation);receipt=folder/'EXECUTION_WALL_WATCHDOG.json'
    atomic_json(receipt,dict(policy='COMMON_WALL_WATCHDOG_V1',command_frames=frames,
        timeout_s=timeout,previous_timeout_s=prior,previous_invocation=before,current_invocation=record(path),
        formula='max(1200, 120 + 2 * complete command frames)',
        simulator_arguments_unchanged=True,command_arrays_unchanged=True,simulated_timing_unchanged=True,
        planner_budget_unchanged=True,task_outcomes_used=False))
    return receipt
