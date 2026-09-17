"""Compact source/goal/measured relation diagnostics for the retained prototype."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json
from .source_phase import COMMON,mean_pose,pose
from .morphology_repair import assign_measured,world_wrist


def run(out):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id'];root=out/'prototype'/sid/'full_task_connection/ead64a3c8365'
    phase=read(out/'source_phase'/sid/'PHASE_RECORD.json');priors=dict(np.load(out/'source_phase'/sid/'SOURCE_PRIORS.npz'))
    contact=read(root/'SELECTED_CONTACTS.json');x=np.asarray(contact['objects']['left']);raw=mean_pose(priors['inferred_object_from_left'][phase['handoff_sample_indices']]);plan=read(root/'physical_plan_advance_90/PLAN.json')
    c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));g.assign(np.asarray(contact['q']));wrist_deviation={}
    for side in ['left','right']:
        prior=mean_pose(priors[side+'_wrist_world'][phase['handoff_sample_indices']]);target=world_wrist(g,side)
        wrist_deviation[side]=dict(position_m=float(np.linalg.norm(target[:3,3]-prior[:3,3])),rotation_rad=float(Rotation.from_matrix(prior[:3,:3].T@target[:3,:3]).magnitude()))
    a=dict(np.load(root/'geometry_telemetry_diagnostic/event_log.npz'));rows=[]
    for stage in ['HOLD_ELEVATED','LEFT_CARRY','LEFT_CARRY_STABILIZE','RECEIVER_APPROACH','DUAL_SUPPORT','GIVER_CLEARANCE']:
        ids=np.flatnonzero(a['stage']==stage)[7::8];values=[]
        for i in ids:
            assign_measured(g,a,int(i));pred={s:world_wrist(g,s)@np.asarray(contact['contacts'][s]['T_wrist_H'])@np.asarray(contact['contacts'][s]['T_HO']) for s in ['left','right']}
            actual=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
            side='right' if stage=='GIVER_CLEARANCE' else 'left';held=pred[side]
            values.append([np.linalg.norm(held[:3,3]-actual[:3,3]),Rotation.from_matrix(held[:3,:3].T@actual[:3,:3]).magnitude(),np.linalg.norm(pred['left'][:3,3]-pred['right'][:3,3]),Rotation.from_matrix(pred['left'][:3,:3].T@pred['right'][:3,:3]).magnitude()])
        v=np.asarray(values)
        rows.append(dict(stage=stage,samples=len(v),held_relation_position_median_m=float(np.median(v[:,0])),held_relation_position_max_m=float(v[:,0].max()),held_relation_rotation_median_rad=float(np.median(v[:,1])),predicted_interhand_object_position_median_m=float(np.median(v[:,2])),predicted_interhand_object_rotation_median_rad=float(np.median(v[:,3]))))
    source_duration=float(priors['source_timestamp'][-1]-priors['source_timestamp'][0]);result=dict(source_id=sid,source_prior=raw,selected_object_goal=x,handoff_relocation_m=float(np.linalg.norm(x[:3,3]-raw[:3,3])),handoff_rotation_adaptation_rad=float(Rotation.from_matrix(raw[:3,:3].T@x[:3,:3]).magnitude()),
        wrist_prior_deviation_at_handoff=wrist_deviation,planned_handoff_object_consistency_m=contact['position_disagreement_m'],planned_handoff_object_consistency_rad=contact['rotation_disagreement_rad'],source_duration_s=source_duration,planned_duration_s=plan['duration_s'],time_stretch=plan['duration_s']/source_duration,planned_phase_times=plan['timing'],
        measured_relations=rows,trace=record(root/'geometry_telemetry_diagnostic/event_log.npz'),contact_selection=record(root/'SELECTED_CONTACTS.json'),
        interpretation='Handoff wrist prior deviations describe spatial adaptation, not dense fidelity or execution gates. Measured relation errors use full articulation FK but inherited incomplete-state contact calibration, explicitly unqualified. Inter-hand consistency outside actual dual support is diagnostic only. These are one-source development measurements, not physical coupling attribution.')
    atomic_json(out/'MECHANISTIC_DIAGNOSTICS.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=run(a.run_dir);print({k:r[k] for k in ['handoff_relocation_m','planned_duration_s','time_stretch']})
