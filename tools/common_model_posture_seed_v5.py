"""A deterministic arm posture derived from G1 link geometry alone."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from tools.common_full_chain_certificate_v5 import full_chain_enclosures

def model_posture_seed(g,natural):
    g.assign(natural);rows=full_chain_enclosures(g);q=natural.copy();m=g.model;proof=[]
    for arm,row in enumerate(rows):
        side=row['side'];ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_'+x+'_link') for x in ('shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
        v1,v2,v3,v4,v5=[m.body_pos[x].copy() for x in ids]
        yaw=row['full_chain_certificate']['maximizer_sample_angle_rad']
        a=Rotation.from_rotvec([0,0,-yaw]).apply(v1)+v2;b=v3+v4
        elbow=np.arctan2(b[2],b[0])-np.arctan2(a[2],a[0])
        u=Rotation.from_rotvec([0,-elbow,0]).apply(a)+b;u=u/np.linalg.norm(u)
        pitch=np.arccos(np.clip(u[0],-1,1));roll=np.arctan2(u[1],-u[2])
        q[arm*7+2:arm*7+6]=[yaw,elbow,roll,pitch]
        proof.append(dict(side=side,yaw=yaw,elbow=elbow,wrist_roll=roll,wrist_pitch=pitch,
                          basis='Model-only relaxed maximum-radius posture; used only as deterministic IK seed, never a forced execution state'))
    return np.clip(q,g.arm_limits[:,0]+1e-7,g.arm_limits[:,1]-1e-7),proof
