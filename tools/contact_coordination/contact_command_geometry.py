"""Revalidate recorded adaptive finger motion; never use outcomes for selection.

Nominal contact ingress and closure have pre-physics certificates. Feedback can
change the finger path. This additional audit checks the actual issued path and
its interiors against the measured scene, and reports failures without erasing
or truncating the physical attempt. It is not a pre-physics path certificate.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .io import atomic_json,record


def audit(folder,out,trace,resolution=.005):
    from .runtime_hulls import Checker
    from .source_phase import COMMON,pose
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    config=load_common_config(COMMON);g=G1Kinematics(config,load_scene(config))
    checker=Checker(g,out,list(trace['joint_names']))
    ix=np.r_[np.flatnonzero(np.diff(trace['control_frame'])!=0),len(trace['control_frame'])-1]
    q=trace['EXECUTED_COMMAND'][ix];position=trace['object_position_world_m'][ix]
    rotation=Rotation.from_quat(trace['object_quaternion_xyzw'][ix])
    commanded_names=set(trace['joint_names'])
    extra_names=trace['all_joint_names'] if 'all_joint_names' in trace else None
    extra_positions=trace['all_measured_q_rad'][ix] if 'all_measured_q_rad' in trace else None
    count=0;bad=0;first=[];maximum=0.
    for frame in range(len(q)):
        previous=max(0,frame-1)
        steps=max(1,int(np.ceil(np.max(abs(q[frame]-q[previous]))/resolution)))
        interpolator=Slerp([0.,1.],Rotation.from_matrix(rotation[[previous,frame]].as_matrix()))
        for fraction in np.arange(1,steps+1)/steps:
            command=(1.-fraction)*q[previous]+fraction*q[frame]
            x=pose(interpolator([fraction]).as_matrix()[0],(1.-fraction)*position[previous]+fraction*position[frame])
            extra=dict(zip(extra_names,(1.-fraction)*extra_positions[previous]+
                fraction*extra_positions[frame],strict=True)) if extra_positions is not None else None
            if extra is not None:extra={name:value for name,value in extra.items() if name not in commanded_names}
            hits=checker.check(command,x,('left','right'),all_joint_state=extra)
            count+=1
            for hit in hits:
                if hit['allowed_contact']:continue
                robot_pair=all(not body.startswith('/') and body!='object' for body in hit['bodies'])
                if robot_pair or hit['depth_m']>.003:
                    bad+=1;maximum=max(maximum,hit['depth_m'])
                    if len(first)<100:first.append(dict(control_frame=frame,edge_fraction=float(fraction),**hit))
    result=dict(status='VALID' if not bad else 'INVALID',command_frames=len(q),checked_states=count,
        checked_edges=max(0,len(q)-1),edge_resolution_rad=resolution,forbidden_count=bad,first_forbidden=first,
        maximum_depth_m=maximum,trace=record(folder/'event_log.npz'),
        scope='Actual feedback contact command path revalidated against measured scene; retrospective physical validation, not pre-physics planning',
        nominal_contact_path_certification_separate=True,failures_do_not_truncate_trace=True)
    path=folder/'ADAPTIVE_COMMAND_GEOMETRY_AUDIT.json';atomic_json(path,result);return path
