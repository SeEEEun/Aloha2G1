"""Separate the open giver's complete projected geometry from the object."""
import numpy as np


def goal(checker,arm,fingers,object_pose,wrist,contact_offset):
    checker.check(np.r_[arm,fingers],object_pose,('left','right'))
    m,d,mj=checker.model,checker.data,checker.mj
    direction=wrist[:3,3]-object_pose[:3,3];direction[2]=0.
    if np.linalg.norm(direction)<1e-9:raise ValueError('Undefined horizontal giver withdrawal direction')
    direction/=np.linalg.norm(direction)
    def vertices(gid):
        mesh=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
        points=m.mesh_vert[start:start+n]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        return checker.g1.model_to_world_position(points)
    hand=[]
    for name,row in checker.rows.items():
        body=row.get('body','')
        if row['kind']=='robot_convex' and body.startswith('left_') and ('hand_' in body or 'wrist_' in body):
            hand.append(vertices(mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name)))
    hand=np.vstack(hand);obj=vertices(checker.geo)
    # Along the away direction, move the furthest inward hand vertex beyond
    # the nearest object support plane. No upward motion loads the support pad.
    distance=max(0.,float(np.max(obj@direction)-np.min(hand@direction)))+contact_offset
    target=wrist.copy();target[:3,3]+=distance*direction
    return target,dict(direction_world=direction,distance_m=distance,contact_offset_m=contact_offset,
        derivation='Complete OPEN giver hull support-plane separation from carried-object hull in horizontal projection. No vertical lifting of residual support. Endpoint and connecting path still require full geometry validation.',
        robot_vertex_count=len(hand),object_vertex_count=len(obj))
