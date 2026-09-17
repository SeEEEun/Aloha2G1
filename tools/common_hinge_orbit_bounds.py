"""Common first-hinge orbit outer certificate, using the actual model chain.

The outer sphere loses the first hinge's fixed axial coordinate. Retaining its
circle orbit produces a union-of-balls enclosure. Distance to that enclosure is
a rigorous position-error lower bound; membership is not a reachability proof.
"""
import numpy as np
import mujoco
from tools.common_grouped_workspace_bounds import grouped_enclosures
from tools.common_g1_position_bounds import residual_lower_bounds


def orbit_enclosures(g1):
    rows=grouped_enclosures(g1);m=g1.model
    for row in rows:
        side=row['side'];root=g1.shoulder_anchor_ids[side]
        child=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_shoulder_roll_link')
        axis=m.jnt_axis[m.body_jntadr[root]];v=m.body_pos[child]
        radius=float(np.linalg.norm(v-axis*(v@axis)))
        rotation=g1.data.xmat[root].reshape(3,3)
        row['orbit']=dict(center_m=row['balls'][-1]['center_m'],axis_model=(rotation@axis).tolist(),
            orbit_radius_m=radius,remainder_radius_m=row['balls'][-1]['radius_m']-radius,
            derivation='The first child translation rotates on a fixed circle under the first hinge. The remaining grouped chain is enclosed in a ball about each point of that circle. Joint limits and collisions can only shrink this outer set.')
    return rows


def orbit_lower_bounds(target,enclosures):
    old=residual_lower_bounds(target,enclosures);values=[]
    for t,row,previous in zip(target,enclosures,old):
        orbit=row['orbit'];axis=np.array(orbit['axis_model']);v=t-orbit['center_m']
        axial=float(v@axis);radial=float(np.linalg.norm(v-axial*axis))
        distance=np.hypot(radial-orbit['orbit_radius_m'],axial)-orbit['remainder_radius_m']
        values.append(max(previous,distance,0.))
    return np.array(values)
