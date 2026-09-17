"""Additional conservative geometry certificates; no target or identity tuning.

Unlike a triangle sum, adjacent translations around a hinge cannot align their
axial components independently. Grouping them gives a tighter OUTER enclosure.
Being inside any of these enclosures still never proves reachability.
"""
import numpy as np
import mujoco
from tools.common_g1_position_bounds import outer_enclosures


def hinge_pair_radius(first,second,axis):
    axis=axis/np.linalg.norm(axis)
    a=float(first@axis);b=float(second@axis)
    # Unrestricted hinge angle is an outer relaxation of authoritative limits.
    return float(np.hypot(a+b,np.linalg.norm(first-a*axis)+np.linalg.norm(second-b*axis)))


def grouped_enclosures(g1):
    result=outer_enclosures(g1);m=g1.model
    for row in result:
        side=row['side']
        def body(name):return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_'+name+'_link')
        yaw,elbow,roll,pitch=[body(s) for s in ('shoulder_yaw','elbow','wrist_roll','wrist_pitch')]
        for b in (yaw,elbow,roll,pitch):np.testing.assert_allclose(m.body_quat[b],[1,0,0,0],atol=1e-15,rtol=0)
        yaw_axis=m.jnt_axis[m.body_jntadr[yaw]];roll_axis=m.jnt_axis[m.body_jntadr[roll]]
        upper=hinge_pair_radius(m.body_pos[yaw],m.body_pos[elbow],yaw_axis)
        wrist=hinge_pair_radius(m.body_pos[roll],m.body_pos[pitch],roll_axis)
        old=float(sum(np.linalg.norm(m.body_pos[b]) for b in (yaw,elbow,roll,pitch)))
        improvement=old-upper-wrist
        assert improvement>=-1e-15
        base=row['balls'][1]
        row['balls'].append(dict(center_m=base['center_m'],radius_m=base['radius_m']-improvement,
            derivation='First-hinge invariant axial center, triangle inequality between groups, exact unrestricted-hinge maximum norm within shoulder-yaw/elbow and wrist-roll/pitch translation groups.',
            grouping_improvement_m=improvement,upper_group_radius_m=upper,wrist_group_radius_m=wrist,
            ignores_collision_and_reduces_no_authoritative_limits=True))
    return result
