"""Robot-relative contact assets from the measured standalone controls."""
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json
from .source_phase import COMMON,pose
from .morphology_repair import assign_measured,world_wrist


def build(out):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    source=out/'target_repair/STANDALONE_CONTACT_CALIBRATION_FULL_STATE.json'
    standalone=read(source);s=standalone['contacts']
    contacts=dict(pickup=s['left_GRAVITY_RETENTION'],left_carry=s['left_HOLD_ELEVATED'],
                  handoff_left=s['left_HOLD_ELEVATED'],handoff_right=s['right_HOLD_ELEVATED'])
    path=out/'common_control/left_full_state/event_log.npz';a=dict(np.load(path))
    c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));tool=np.asarray(contacts['pickup']['T_wrist_H'])
    initial=pose(Rotation.from_quat(a['object_quaternion_xyzw'][0]).as_matrix(),a['object_position_world_m'][0])
    ids=np.flatnonzero(a['stage']=='GRAVITY_RETENTION')
    for name,i in [('acquisition_intent',int(ids[len(ids)//2])),('pregrasp',0)]:
        assign_measured(g,a,i,'EXECUTED_COMMAND')
        contacts[name]=dict(side='left',T_HO=np.linalg.inv(world_wrist(g,'left')@tool)@initial,
            T_wrist_H=tool,seed_q=a['MEASURED_Q'][i,:14],
            measured_finger_q=a['MEASURED_Q'][i,14:28],commanded_finger_q=a['EXECUTED_COMMAND'][i,14:28],
            evidence='CALIBRATED_COMMAND_RELATIVE_TO_INITIAL_SUPPORTED_OBJECT',rows=[i],control=record(path),
            full_named_articulation_used=True)
    result=dict(schema='full_state_standalone_contact_calibration_v1',contacts=contacts,
        joint_names=standalone['joint_names'],source=record(source),
        transform_convention='T_AB maps B into A',
        use='Qualified robot-relative single-hand contact transforms and IK seeds only; source determines scene and task-space priors.',
        handoff_pair_physical_qualification='NOT_YET_DEMONSTRATED',
        full_scripted_world_path_used=False)
    atomic_json(out/'target_repair/CONTACT_CALIBRATION.json',result)
    return result


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    build(p.parse_args().run_dir)
