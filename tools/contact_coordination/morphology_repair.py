"""TRAIN repair: separate source relations from calibrated target morphology.

Only contact frames and deterministic IK seeds are extracted from calibration.
No calibration arm trajectory or world task pose is exported as a source plan.
T_AB maps B coordinates into A. H is a fixed functional hand frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT, read, record, atomic_json, atomic_npz
from .source_phase import COMMON, pose, mean_pose

PREVIOUS = ROOT/'outputs/contact_coordination_hybrid/20260907T073437Z'
CONTROL = PREVIOUS/'common_control/scripted_captured/event_log.npz'
STANDALONE = ROOT/'outputs/contact_coordination_retargeting/20260907T063704Z/common_control/left_d42c796d7e59/event_log.npz'


def world_wrist(g1, side):
    t=g1.wrist_pose(side)
    return pose(g1.model_to_world_rotation(t[:3,:3]),g1.model_to_world_position(t[:3,3]))


def assign_named(g1,q,names):
    values=dict(zip(names,q,strict=True))
    g1.assign(np.asarray(q[:14]),*[np.asarray([values[n] for n in g1.hand_joint_names[s]]) for s in ('left','right')])


def assign_measured(g1,trace,index,arm_hand_field='MEASURED_Q'):
    """Preserve measured waist/leg states when reconstructing calibration FK."""
    import mujoco
    if not {'all_measured_q_rad','all_joint_names'} <= trace.keys():
        raise ValueError('CALIBRATION_RECAPTURE_REQUIRED: all_measured_q_rad and all_joint_names; 28D traces omit loaded waist deflection')
    assign_named(g1,trace[arm_hand_field][index],trace['joint_names'])
    controlled=set(trace['joint_names'])
    for name,value in zip(trace['all_joint_names'],trace['all_measured_q_rad'][index],strict=True):
        if name in controlled:continue
        joint=mujoco.mj_name2id(g1.model,mujoco.mjtObj.mjOBJ_JOINT,str(name))
        if joint<0:raise ValueError('Runtime joint missing from FK: '+str(name))
        g1.data.qpos[g1.model.jnt_qposadr[joint]]=value
    mujoco.mj_forward(g1.model,g1.data)


def extract_calibration(g1,control_path=CONTROL):
    """Predeclared stage medians; fixed reference hand frames derived once."""
    a=dict(np.load(control_path)); names=a['joint_names'].astype(str).tolist()
    if not {'all_measured_q_rad','all_joint_names'} <= a.keys():
        raise ValueError('CALIBRATION_RECAPTURE_REQUIRED: full named measured articulation states are missing from '+str(control_path))
    specs=[('pickup','GRAVITY_RETENTION','left'),('left_carry','HOLD_ELEVATED','left'),
           ('handoff_left','LEFT_HANDOFF_HOLD','left'),('handoff_right','RIGHT_POST_RELEASE_RETENTION','right')]
    result={}
    # Define fixed wrist->H using the qualified measured HOLD morphology.
    tools={}
    for side,stage in [('left','HOLD_ELEVATED'),('right','RIGHT_POST_RELEASE_RETENTION')]:
        ids=np.flatnonzero(a['stage']==stage); i=int(ids[len(ids)//2]);q=a['MEASURED_Q'][i]
        assign_measured(g1,a,i); tools[side]=np.linalg.inv(g1.wrist_pose(side))@g1.whole_hand_grasp_pose(side)
    for name,stage,side in specs:
        ids=np.flatnonzero(a['stage']==stage); ids=ids[len(ids)//3:2*len(ids)//3:8]
        values=[]
        for i in ids:
            assign_measured(g1,a,i)
            x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
            values.append(np.linalg.inv(world_wrist(g1,side)@tools[side])@x)
        i=int(ids[len(ids)//2]);g=mean_pose(values)
        result[name]=dict(side=side,T_HO=g,T_wrist_H=tools[side],seed_q=a['MEASURED_Q'][i,:14],
            measured_finger_q=a['MEASURED_Q'][i,14:28],commanded_finger_q=a['EXECUTED_COMMAND'][i,14:28],
            evidence='MEASURED_CALIBRATION_CONTACT',stage=stage,rows=ids,
            relation_position_spread_m=float(np.max(np.linalg.norm(np.asarray(values)[:,:3,3]-g[:3,3],axis=1))))
    # Acquisition is a calibrated approach relative to the INITIAL supported
    # object, not a post-lift object estimate. This keeps calibrated pushing.
    ids=np.flatnonzero(a['stage']=='GRAVITY_RETENTION');i=int(ids[len(ids)//2])
    assign_measured(g1,a,i,'EXECUTED_COMMAND')
    x_initial=pose(Rotation.from_quat(a['object_quaternion_xyzw'][0]).as_matrix(),a['object_position_world_m'][0])
    result['acquisition_intent']=dict(side='left',T_HO=np.linalg.inv(world_wrist(g1,'left')@tools['left'])@x_initial,
        T_wrist_H=tools['left'],seed_q=a['MEASURED_Q'][i,:14],
        measured_finger_q=a['MEASURED_Q'][i,14:28],commanded_finger_q=a['EXECUTED_COMMAND'][i,14:28],
        evidence='CALIBRATED_INITIAL_OBJECT_TO_ACQUISITION_COMMAND',stage='GRAVITY_RETENTION',rows=[i],
        physical_capture_displacement_m=a['object_position_world_m'][i]-a['object_position_world_m'][0])
    assign_measured(g1,a,0,'EXECUTED_COMMAND')
    result['pregrasp']=dict(side='left',T_HO=np.linalg.inv(world_wrist(g1,'left')@tools['left'])@x_initial,
        T_wrist_H=tools['left'],seed_q=a['MEASURED_Q'][0,:14],
        measured_finger_q=a['MEASURED_Q'][0,14:28],commanded_finger_q=a['EXECUTED_COMMAND'][0,14:28],
        evidence='CALIBRATED_PREGRASP_RELATIVE_TO_INITIAL_OBJECT',stage='LEFT_OPEN',rows=[0])
    return dict(schema='target_morphology_contact_calibration_v1',transform_convention='T_AB maps B into A',
                control=record(control_path),joint_names=names,contacts=result,
                use='contact morphology and IK seeds only; source defines scene and phase goals')


def wrist_target(object_pose, contact):
    return np.asarray(object_pose)@np.linalg.inv(contact['T_HO'])@np.linalg.inv(contact['T_wrist_H'])


def run(out):
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.audit_final_grasp_capture_task_frames import DistanceModel
    from .planner import realize_phase_goals
    from .prototype import INITIAL
    common=load_common_config(COMMON);g1=G1Kinematics(common,load_scene(common))
    cal=extract_calibration(g1);atomic_json(out/'target_repair/CONTACT_CALIBRATION.json',cal)
    sid=read(out/'bootstrap/SELECTION.json')['prototype_source_id'];phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
    x=np.asarray(phase['initial_object_pose_world']); contact=cal['contacts']['acquisition_intent']
    target=wrist_target(x,contact);natural=np.asarray(read(INITIAL)['g1_14_arm_initial_q_rad'])
    physics=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')
    dims=np.asarray(next(r for r in physics['geometry_candidates'] if r['name']=='INTERMEDIATE_PLUSH_PROXY')['dimensions_m'])
    dm=DistanceModel(g1,dims,.795,tuple(cal['joint_names']));dm.set_object_pose(x,-(.085-dims[2])/2)
    old=read(PREVIOUS/'prototype'/sid/'INTERACTION_OURS/aefa49031c82/PHASE_IK.json')
    geometry=[]
    for row in old['phases'][:3]:
        q=np.asarray(row['candidates'][row['selected_seed']]['q']); full=np.r_[q,contact['commanded_finger_q']]
        dm.assign(full); assign_named(g1,full,cal['joint_names'])
        model_geom=g1.trajectory_geometry(q[None],*[np.asarray([[dict(zip(cal['joint_names'],full))[n] for n in g1.hand_joint_names[s]]]) for s in ('left','right')],1e-5)
        geometry.append(dict(phase=row['phase'],q=full,object_pose_world=x,wrists={s:world_wrist(g1,s) for s in ('left','right')},
            digit_object_distances_m=dm.distances(),robot_collisions=model_geom['collision_records'],
            limitation='MuJoCo primitive diagnostic; endpoint only; no runtime parity certificate'))
    atomic_json(out/'failure_localization/OLD_EARLY_PHASE_GEOMETRY.json',geometry)
    goals=[]
    for name,zoffset in [('PREGRASP_CLEARANCE',.085),('LEFT_ACQUISITION',0.),('LIFT',.063)]:
        t=target.copy();t[2,3]+=zoffset
        goals.append(dict(name=name,active_hands=['left'],wrist_pose_world={'left':t},position_tolerance_m=.003,orientation_tolerance_rad=.05,
            source_id=sid,source_initial_object_pose=x,contact_calibration='acquisition_intent',
            offset_provenance='existing visual object height / qualified lift clearance'))
    config=dict(position_residual_scale=100.,orientation_residual_scale=1.,joint_prior_scale=.001,max_nfev_per_seed_per_goal=120)
    # Qualified posture is an additional deterministic seed shared by methods.
    results=[]
    for name,seed in [('natural',natural),('qualified_contact',np.asarray(contact['seed_q']))]:
        r=realize_phase_goals(g1,goals,seed,config);q=r.pop('q');atomic_npz(out/f'target_repair/{name}_IK.npz',q=q)
        r['seed_name']=name
        for row,state in zip(r['phases'],q[1:]):
            full=np.r_[state,contact['commanded_finger_q']];dm.assign(full)
            row['digit_object_distances_m_at_initial_object']=dm.distances()
        results.append(r)
    atomic_json(out/'target_repair/GOALS.json',goals);atomic_json(out/'target_repair/IK_RESULT.json',results)
    return dict(status='TARGET_MORPHOLOGY_REPAIR_DIAGNOSTIC',results=results,old_geometry=geometry)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    run(args.run_dir)
