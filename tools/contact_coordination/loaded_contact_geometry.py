"""Local loaded-state prediction for common offline contact geometry.

Joint targets are servo setpoints. In the calibrated retained phase, using
targets as measured arm positions creates a documented false object overlap.
The model predicts named arm deflection; it never changes runtime commands,
colliders, measurements, or the independent physical scorer.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json,fingerprint


def calibrate(out,calibration):
    c=read(calibration)['contacts']['right_carry_command_intent']
    trace=c['control'];assert record(trace['path'])==trace
    a=np.load(trace['path']);ids=np.asarray(c['rows'],dtype=int)
    delta=a['MEASURED_Q'][ids,:14]-a['EXECUTED_COMMAND'][ids,:14]
    controlled=set(a['joint_names'].tolist())
    extra={str(name):float(a['all_measured_q_rad'][ids,j].mean()) for j,name in enumerate(a['all_joint_names']) if name not in controlled}
    result=dict(joint_names=a['joint_names'][:14].tolist(),arm_measured_minus_command_rad=delta.mean(axis=0),
        schema_version=2,uncommanded_joint_positions_rad=extra,
        observed_bias_min_rad=delta.min(axis=0),observed_bias_max_rad=delta.max(axis=0),
        measured_seed_q=a['MEASURED_Q'][ids[len(ids)//2],:14],command_seed_q=a['EXECUTED_COMMAND'][ids[len(ids)//2],:14],
        T_wrist_H=c['T_wrist_H'],measured_T_HO=c['measured_T_HO'],command_T_HO=c['T_HO'],
        trace=trace,rows=ids,calibration=record(calibration),implementation=record(__file__),
        application='Common offline expected loaded geometry in sole-right carry with open stationary giver. Raw command robot/robot and environment checks are retained. Object interaction uses predicted loaded arms and phase-calibrated fingers.',
        limitation='Local quasistatic TRAIN calibration, not a global drive model or a physical validity certificate. Actual measured rollout geometry, contacts, and raw limits remain mandatory.',
        runtime_commands_or_physics_changed=False,source_world_trajectory_copied=False)
    key=fingerprint([__file__,calibration,trace['path']])[0]
    path=out/'target_repair'/('LOADED_CARRY_GEOMETRY_'+key[:12]+'.json')
    if path.exists():assert read(path)==__import__('json').loads(__import__('json').dumps(result,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v))
    else:atomic_json(path,result)
    return path


def predicted_q(q,model,alpha=1.):
    p=np.asarray(q).copy();p[:14]+=float(alpha)*np.asarray(model['arm_measured_minus_command_rad']);return p


def kinematic_calibration(model):
    return dict(arm_joint_offset_rad=model['arm_measured_minus_command_rad'],
                uncommanded_joint_positions_rad=model.get('uncommanded_joint_positions_rad',{}))


def object_pose(g1,q,model):
    from .planner import assign_kinematic_state
    from .morphology_repair import world_wrist
    assign_kinematic_state(g1,np.asarray(q)[:14],kinematic_calibration(model))
    return world_wrist(g1,'right')@np.asarray(model['T_wrist_H'])@np.asarray(model['measured_T_HO'])


def check(checker,q,x,allowed,model,alpha=1.,verify_relation=True,audit=None):
    from .morphology_repair import world_wrist
    raw=getattr(checker,'query',checker.check)(q,x,allowed,object_environment=True)
    # Preserve raw-target robot/robot and environment safety. The object must
    # be checked against the predicted physical arm pose, not an unloaded
    # setpoint combined with loaded finger/object coordinates.
    hits=[dict(prediction='RAW_COMMAND_NONOBJECT',**h) for h in raw if 'hybrid_object' not in h['geoms']]
    p=predicted_q(q,model,alpha)
    extra=model.get('uncommanded_joint_positions_rad',{})
    if extra:
        # Explicit full-articulation prediction, never a runtime state write.
        hits.extend(dict(prediction='CALIBRATED_LOADED_CONTACT',**h) for h in getattr(checker,'query',checker.check)(p,x,allowed,all_joint_state=extra,object_environment=True))
    else:hits.extend(dict(prediction='CALIBRATED_LOADED_CONTACT',**h) for h in getattr(checker,'query',checker.check)(p,x,allowed,object_environment=True))
    limit_excess=float(np.maximum(np.maximum(checker.g1.arm_limits[:,0]-p[:14],p[:14]-checker.g1.arm_limits[:,1]),0).max())
    if limit_excess>0:
        hits.append(dict(kind='PREDICTED_MEASURED_ARM_LIMIT_VIOLATION',allowed_contact=False,excess_rad=limit_excess))
    dp=dr=None
    if verify_relation and alpha==1.:
        held=object_pose(checker.g1,q,model)
        dp=float(np.linalg.norm(held[:3,3]-x[:3,3]));dr=float(Rotation.from_matrix(held[:3,:3].T@x[:3,:3]).magnitude())
        if dp>.003 or dr>.05:
            hits.append(dict(kind='LOADED_CARRY_RELATION_OUTSIDE_EXISTING_GOAL_TOLERANCE',allowed_contact=False,
                             position_m=dp,orientation_rad=dr,prediction='CALIBRATED_LOADED_CONTACT'))
    if audit is not None:audit.append(dict(alpha=alpha,raw_command_object_overlaps=[h for h in raw if 'hybrid_object' in h['geoms'] and not h['allowed_contact']],
        predicted_position_error_m=dp,predicted_orientation_error_rad=dr,predicted_arm_limit_excess_rad=limit_excess,
        maximum_arm_bias_rad=float(np.max(np.abs(p[:14]-q[:14])))))
    return hits


def departure_region(checker,q,fingers,x,wrist,model,contact_offset):
    """Three geometry-derived directions; no successful world waypoint."""
    checker.check(predicted_q(np.r_[q,fingers],model),x,('right',),all_joint_state=model.get('uncommanded_joint_positions_rad') or None)
    m,d,mj=checker.model,checker.data,checker.mj
    def vertices(gid):
        mesh=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
        p=m.mesh_vert[start:start+n]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        return checker.g1.model_to_world_position(p)
    hand=[]
    for name,row in checker.rows.items():
        body=row.get('body','')
        if row['kind']=='robot_convex' and body.startswith('left_') and ('hand_' in body or 'wrist_' in body):
            hand.append(vertices(mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name)))
    hand=np.vstack(hand);obj=vertices(checker.geo)
    away=x[:3,3]-hand.mean(axis=0);away[2]=0.;away/=np.linalg.norm(away)
    shoulder=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'right_shoulder_pitch_link')
    if shoulder<0:raise ValueError('Named receiver shoulder missing from runtime model')
    toward=checker.g1.model_to_world_position(d.xpos[shoulder])-wrist[:3,3];toward/=np.linalg.norm(toward)
    lateral=toward.copy();lateral[2]=0.;lateral/=np.linalg.norm(lateral);lateral+=away;lateral/=np.linalg.norm(lateral)
    directions=[toward,lateral,np.array([0.,0.,1.])]
    # A separating support plane is sufficient, but not necessary for a valid
    # departure. Its158mm vertical displacement exceeded this arm's local
    # workspace. Take a small object-scale step and check actual full geometry.
    short_extent=float(np.min(np.ptp(obj@x[:3,:3],axis=0)))
    goals=[]
    for fraction in (.5,1.):
        for direction in directions:
            distance=fraction*short_extent
            xp=x.copy();xp[:3,3]+=distance*direction;target=wrist.copy();target[:3,3]+=distance*direction
            goals.append(dict(name='RECEIVER_DEPARTURE',active_hands=['right'],wrist_pose_world={'right':target},
                contact_mode='RIGHT_HOLD',object_pose=xp,position_tolerance_m=.003,orientation_tolerance_rad=.05,
                clearance_geometry=dict(direction_world=direction,distance_m=distance,contact_offset_m=contact_offset,
                    rule='Half/full short-object-extent step toward receiver shoulder, its combination with away-from-giver direction, or up. These are candidate connections, not reachability certificates. Preserve full wrist/object orientation and check every edge/loaded relation.')))
    return goals


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);a=p.parse_args();print(calibrate(a.run_dir.resolve(),a.calibration.resolve()))
