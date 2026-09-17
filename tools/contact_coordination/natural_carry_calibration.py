"""Measure a shared object-in-hand primitive after natural acquisition/lift.

This stores robot-relative contact morphology and an IK seed, never a world
trajectory. The calibrated contact is shared by both representations.
"""
from pathlib import Path
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json
from .source_phase import COMMON,pose,mean_pose
from .morphology_repair import assign_measured,world_wrist


def build(out,trace_path):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.finalize_common_dex3_grasp_qualification import CONTACT_THRESHOLD_N,TABLE_THRESHOLD_N
    path=out/'target_repair/CONTACT_CALIBRATION.json';cal=read(path)
    previous=out/'target_repair/CONTACT_CALIBRATION_BEFORE_NATURAL_CARRY.json'
    if previous.exists():raise FileExistsError(previous)
    atomic_json(previous,cal)
    a=dict(np.load(trace_path));cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
    ids=np.flatnonzero(a['stage']=='HOLD_ELEVATED')
    assert len(ids)>=240,'Full1s retained contact evidence required'
    selected=ids[len(ids)//3:2*len(ids)//3:8]
    mask=(a['left_thumb_force_n'][ids]>=CONTACT_THRESHOLD_N)&((a['left_index_force_n'][ids]>=CONTACT_THRESHOLD_N)|(a['left_middle_force_n'][ids]>=CONTACT_THRESHOLD_N))&(a['table_contact_force_n'][ids]<=TABLE_THRESHOLD_N)
    assert np.all(mask),'Use sustained measured airborne opposing contact only'
    fixed=np.asarray(cal['contacts']['left_carry']['T_wrist_H']);relations=[]
    for i in selected:
        assign_measured(g,a,i);x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
        relations.append(np.linalg.inv(world_wrist(g,'left')@fixed)@x)
    relation=mean_pose(relations);i=int(selected[len(selected)//2]);old=np.asarray(cal['contacts']['left_carry']['T_HO'])
    contact=copy.deepcopy(cal['contacts']['left_carry'])
    contact.update(T_HO=relation,measured_finger_q=a['MEASURED_Q'][i,14:28],commanded_finger_q=a['EXECUTED_COMMAND'][i,14:28],seed_q=a['MEASURED_Q'][i,:14],rows=selected,
        stage='HOLD_ELEVATED',evidence='MEASURED_NATURAL_START_ACQUISITION_LIFT_RETENTION',trace=record(trace_path),
        relation_position_spread_m=float(np.max(np.linalg.norm(np.asarray(relations)[:,:3,3]-relation[:3,3],axis=1))),
        relation_rotation_spread_rad=float(np.max(Rotation.from_matrix(np.asarray(relations)[:,:3,:3]@relation[:3,:3].T).magnitude())))
    cal['contacts']['left_carry']=contact;cal['contacts']['handoff_left']=copy.deepcopy(contact)
    cal['natural_carry_calibration']=dict(before=record(previous),trace=record(trace_path),
        object_in_hand_position_change_m=float(np.linalg.norm(old[:3,3]-relation[:3,3])),object_in_hand_rotation_change_rad=float(Rotation.from_matrix(old[:3,:3].T@relation[:3,:3]).magnitude()),
        cause='Pregrasp-initialized standalone contact differs from the contact realized after the shared natural approach. Use measured post-lift natural-acquisition morphology for later carry/handoff phases.',
        shared_A_B_asset=True,world_task_trajectory_copied=False,source_id_branch=False,controller_changed=False)
    atomic_json(path,cal);return cal['natural_carry_calibration']


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);a=p.parse_args();print(build(a.run_dir.resolve(),a.trace.resolve()))
