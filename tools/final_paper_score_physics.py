#!/usr/bin/env python3
"""Cumulative stages from measured physics, with no method-specific thresholds."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_reference_physics import STAGES,DEST,read,atomic_json,file_record
from tools.cartesian_reachability_forensic import model

def score(folder):
    invocation=read(folder/'INVOCATION_MANIFEST.json');key=Path(invocation['commands']['path']).stem
    row=next(r for r in read(DEST/'CASE_MANIFEST.json') if r['key']==key)
    existing=folder/'EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json'
    if not existing.exists():
        subprocess.run([sys.executable,str(ROOT/'tools/score_episode_registered_physical_eval35_run.py'),'--run-dir',str(folder)],cwd=ROOT,check=False)
    old=read(existing)
    with np.load(folder/'event_log.npz') as z:
        events=z['DIRECT_COMMON_EXECUTION_EVENTS'].astype(str);frames=z['control_frame'].astype(int)
        measured=z['MEASURED_Q'];positions=z['object_position_world_m'];table=z['table_contact_force_n']
    with np.load(invocation['commands']['path']) as z:clock=dict(zip(z['source_event_names'].astype(str),z['execution_event_frames'].astype(int)))
    indices=np.r_[np.flatnonzero(np.diff(frames)!=0),len(frames)-1]
    # Approach uses measured FK-to-source-conditioned initial object distance,
    # not a learned graspability atlas or a method's intended wrist target.
    g,c,n=model();initial_object=positions[0];distances=[];control=frames[indices]
    for v in measured[indices]:
        wrist=g.wrist_state(v[:14])['left_position'];world=g.model_to_world_position(wrist)
        distances.append(float(np.linalg.norm(world-initial_object)))
    distances=np.array(distances);scope=(control>=clock['APPROACH_START'])&(control<=clock['LEFT_CLOSE_BEGIN'])
    approach=bool(scope.any() and distances[scope].min()<distances[0]-1e-6)
    confirm=np.flatnonzero(np.char.find(events,'LEFT_GRASP_CONFIRMED')>=0)
    lifted=np.flatnonzero(np.char.find(events,'LEFT_GRASP_STATE_LIFT')>=0)
    grasp=bool(len(confirm) and frames[confirm[0]]<clock['LEFT_LIFT_BEGIN'])
    lift=bool(grasp and len(lifted) and old['outcomes']['LEFT_GRASP_SUCCESS'])
    independent=[True,approach,grasp,lift,old['outcomes']['HANDOFF_SUCCESS'],old['outcomes']['RIGHT_OWNERSHIP_SUCCESS'],old['outcomes']['NO_DROP_TO_BIN'],old['outcomes']['BIN_ENTRY_SUCCESS'],old['outcomes']['BIN_SETTLE_SUCCESS'],old['outcomes']['FULL_TASK_SUCCESS']]
    cumulative=np.logical_and.accumulate(independent).tolist();outcomes=dict(zip(STAGES,cumulative))
    full=outcomes['FULL_TASK_SUCCESS'];first=next((s for s,v in outcomes.items() if not v),'SUCCESS')
    r=dict(case=row,outcome='PHYSICAL_SUCCESS' if full else 'PHYSICAL_FAILURE',outcomes=outcomes,
        independent_diagnostic_stages=dict(zip(STAGES,independent)),first_failure_stage=first,
        complete_trace=True,physical_trace=file_record(folder/'event_log.npz'),legacy_scorer=file_record(existing),
        integrity=old['integrity'],source_clock=clock,freeze=invocation['freeze'],
        approach_initial_distance_m=float(distances[0]),approach_min_distance_m=float(distances[scope].min()) if scope.any() else None,
        physical_validity=old['integrity']['hard_physical_validity'],denominator=35,
        note='Normal measured physical failures are data, not automatically rerunnable infrastructure faults.')
    atomic_json(folder/'RESULT.json',r);print(json.dumps(r,indent=2))

if __name__=='__main__':score(Path(sys.argv[1]).resolve())
