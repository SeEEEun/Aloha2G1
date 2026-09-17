"""Finite method-blind position-nullspace search over terminal wrist hinges.

Constant hinge trajectories cannot create temporal jumps. Proxies rank seeds
only; detailed geometry alone decides physical validity. The wrist position is
verified unchanged, and no collision pair is waived.
"""
from itertools import product
import numpy as np


def rank_constant_nullspace(kinematics,collision,q,hands):
    joints=[6,13]
    # Derive/verify the positional nullspace from the compiled kinematic tree.
    for side,index in zip(('left','right'),joints):
        body=kinematics.wrist_ids[side];jid=kinematics.arm_joint_ids[index]
        assert kinematics.model.jnt_bodyid[jid]==body
        np.testing.assert_array_equal(kinematics.model.jnt_pos[jid],np.zeros(3))
    lo=kinematics.arm_limits[joints,0];hi=kinematics.arm_limits[joints,1]
    levels=(np.arange(7)+.5)/7
    rows=[]
    for i,j in product(range(7),repeat=2):
        values=lo+(hi-lo)*np.array([levels[i],levels[j]])
        candidate=q.copy();candidate[:,joints]=values
        count=0;depth=0.;maximum=0.
        for v,h in zip(candidate,hands):
            rr=collision.proxy._records(v,*h)
            count+=bool(rr)
            for r in rr:
                d=max(r['penetration_depth_m'],0.);depth+=d;maximum=max(maximum,d)
        rows.append(dict(seed_indices=[i,j],wrist_joint_values=values.tolist(),proxy_positive_frames=count,
            proxy_depth_sum_m=depth,maximum_proxy_depth_m=maximum,
            posture_change_norm=float(np.linalg.norm(candidate-q))))
    return sorted(rows,key=lambda r:(r['proxy_positive_frames'],r['proxy_depth_sum_m'],r['posture_change_norm']))
