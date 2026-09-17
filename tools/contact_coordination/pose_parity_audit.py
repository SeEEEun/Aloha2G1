"""Compare saved runtime rigid-body poses with the reused offline named FK.

Read-only diagnostic. Direct convex pair queries avoid treating sparse contact
separation reports as a complete solid-overlap certificate.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json
from .source_phase import COMMON,pose
from .runtime_hulls import Checker


def audit(out,folder):
    import mujoco
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    a=dict(np.load(folder/'event_log.npz'));c=load_common_config(COMMON);g=G1Kinematics(c,load_scene(c));ch=Checker(g,out,list(a['joint_names']))
    body_names=list(a['body_names']);joint_names=list(a['all_joint_names'])
    selected=[0,225,300,540,750,800,832,900,940,950,960,970,980,1016]
    rows=[];m=ch.model;d=ch.data
    for f in selected:
        ids=np.flatnonzero(a['control_frame']==f)
        if not len(ids):continue
        i=int(ids[-1]);x=pose(Rotation.from_quat(a['object_quaternion_xyzw'][i]).as_matrix(),a['object_position_world_m'][i])
        ch.check(a['MEASURED_Q'][i],x,('left','right'));errors={};pairs=[]
        for name in body_names:
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
            if bid<0:continue
            j=body_names.index(name);actual=pose(Rotation.from_quat(a['body_quaternion_xyzw'][i,j]).as_matrix(),a['body_position_world_m'][i,j])
            pred=pose(g.model_to_world_rotation(d.xmat[bid].reshape(3,3)),g.model_to_world_position(d.xpos[bid]))
            errors[name]=dict(position_m=float(np.linalg.norm(pred[:3,3]-actual[:3,3])),rotation_rad=float(Rotation.from_matrix(pred[:3,:3].T@actual[:3,:3]).magnitude()),predicted=pred,actual=actual)
        # Query exactly the same convex geometry using the actual rigid-body
        # transforms. mesh principal-axis/recentering is in model.geom_*.
        for key,row in ch.rows.items():
            if row['kind']!='robot_convex' or row['body'] not in errors:continue
            geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,key);actual=np.asarray(errors[row['body']]['actual']);rot=g.root_pose[:3,:3].T@actual[:3,:3];pos=g.world_to_model_position(actual[:3,3])
            before=float(mujoco.mj_geomDistance(m,d,geom,ch.geo,.03,None))
            d.geom_xpos[geom]=pos+rot@m.geom_pos[geom]
            d.geom_xmat[geom]=(rot@Rotation.from_quat(m.geom_quat[geom][[1,2,3,0]]).as_matrix()).reshape(-1)
            after=float(mujoco.mj_geomDistance(m,d,geom,ch.geo,.03,None))
            if min(before,after)<.003:pairs.append(dict(body=row['body'],offline_distance_m=before,runtime_pose_distance_m=after))
        extra={n:float(a['all_measured_q_rad'][i,j]) for j,n in enumerate(joint_names) if n not in a['joint_names']}
        ch.check(a['MEASURED_Q'][i],x,('left','right'),dict(zip(joint_names,a['all_measured_q_rad'][i],strict=True)))
        corrected={}
        for name,error in errors.items():
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);actual=np.asarray(error['actual'])
            pred=pose(g.model_to_world_rotation(d.xmat[bid].reshape(3,3)),g.model_to_world_position(d.xpos[bid]))
            corrected[name]=dict(position_m=float(np.linalg.norm(pred[:3,3]-actual[:3,3])),rotation_rad=float(Rotation.from_matrix(pred[:3,:3].T@actual[:3,:3]).magnitude()))
        rows.append(dict(frame=f,phase=str(a['stage'][i]),body_errors=errors,object_pair_distances=pairs,extra_joints=extra,full_named_fk_errors=corrected))
        print('POSE_PARITY',f,{n:round(errors[n]['position_m']*1000,3) for n in ('left_wrist_yaw_link','right_wrist_yaw_link') if n in errors},'object_pairs',pairs,flush=True)
    result=dict(trace=record(folder/'event_log.npz'),runtime_geometry=record(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json'),audit=record(__file__),rows=rows,changes_physics=False,
        maximum_full_named_fk_position_error_m=max(e['position_m'] for r in rows for e in r['full_named_fk_errors'].values()),
        maximum_full_named_fk_rotation_error_rad=max(e['rotation_rad'] for r in rows for e in r['full_named_fk_errors'].values()))
    atomic_json(folder/'POSE_PARITY_AUDIT.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--attempt-dir',type=Path,required=True);args=p.parse_args();audit(args.run_dir,args.attempt_dir)
