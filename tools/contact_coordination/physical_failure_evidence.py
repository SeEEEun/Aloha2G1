"""Failure labels and times derived only from retained measured evidence."""
from pathlib import Path
import numpy as np
from .io import read,record


def failure_events(folder,recording,outcome,frames,trace=None):
    folder=Path(folder);events=[]
    protection=folder/'PREGRASP_OBJECT_PROTECTION.json'
    if protection.exists():
        value=read(protection);first=value.get('first_premature_contact')
        rejected=value.get('incidental_contact_assessment',{}).get('rejected_contacts',[])
        if rejected:first=rejected[0]
        if first and value['status']=='PREGRASP_OBJECT_CONTACT':events.append(dict(reason='PREGRASP_OBJECT_CONTACT: '+first['robot_link'],
            control_frame=first['control_frame'],causal_label='PREGRASP_OBJECT_CONTACT',evidence=record(protection)))
        elif value['status']=='PREGRASP_OBJECT_CONTACT':events.append(dict(reason='PREACQUISITION_OBJECT_MOTION_POLICY',
            control_frame=value['acquisition_begin_frame'],causal_label='PREGRASP_OBJECT_CONTACT',evidence=record(protection)))
    task=recording['first_failure_latched']
    if task:
        reason=task['reason']
        label=('PHYSICS_GRASP_FAIL' if 'ACQUISITION' in reason else
            'PHYSICS_OWNERSHIP_FAIL' if 'OWNERSHIP' in reason else 'PHYSICS_HANDOFF_FAIL')
        events.append(dict(task,causal_label=label,evidence=record(folder/'FULL_ATTEMPT_RECORDING.json')))
    geometry=folder/'MEASURED_GEOMETRY_AUDIT.json'
    if geometry.exists():
        hits=read(geometry)['first_forbidden']
        if hits:
            events.append(dict(reason='MEASURED COLLISION: '+' / '.join(hits[0]['bodies']),
                control_frame=int(hits[0]['frame']),causal_label='PHYSICS_FAILURE',evidence=record(geometry)))
    finite_validity=folder/'FINITE_STATE_VALIDITY_TELEMETRY.json'
    if finite_validity.exists():
        for name,check in read(finite_validity)['checks'].items():
            first=check['first_exceedance']
            if first:
                unit='rad/s' if name=='angular_speed' else 'm'
                events.append(dict(reason=f'FINITE {name.upper()} > {check["threshold"]:g} {unit}',
                    control_frame=first['control_frame'],causal_label='PHYSICS_FAILURE',evidence=record(finite_validity)))
    legacy=folder/'LEGACY_CONTROL_SCORER.json'
    if legacy.exists():
        values=read(legacy);tolerance=values.get('penetration_tolerance_m')
        if tolerance is not None and values.get('maximum_doll_bin_penetration_m',0)>tolerance:
            archive=trace if trace is not None else np.load(folder/'event_log.npz',allow_pickle=False)
            try:
                failures=np.flatnonzero(archive['maximum_doll_bin_penetration_m']>tolerance)
                if len(failures):
                    index=int(failures[0])
                    events.append(dict(reason=f'OBJECT-BIN PENETRATION > {tolerance*1000:g} mm',
                        control_frame=int(archive['control_frame'][index]),causal_label='PHYSICS_PLACE_FAIL',
                        evidence=record(legacy)))
            finally:
                if trace is None:archive.close()
    if not events and outcome and not outcome['stages']['FULL_TASK']:
        # An aggregate diagnosis has no evidenced earlier timestamp.
        reason='FINAL PHYSICAL VALIDITY AUDIT' if not outcome['physical_validity'] else 'FINAL TASK SCORE: '+str(outcome.get('first_failed_stage'))
        events.append(dict(reason=reason,control_frame=frames-1,causal_label='PHYSICS_FAILURE',
            evidence=record(folder/'ABC_NOMINAL_SCORE.json')))
    return sorted(events,key=lambda event:event['control_frame'])


def recorded_failures(folder,recording,outcome,frames,trace=None):
    events=failure_events(folder,recording,outcome,frames,trace)
    return (events[0] if events else None),recording['first_failure_latched']
