"""Conservative outer workspace certificates, independent of any IK outcome.

Outside an outer enclosure proves non-reachability; inside proves nothing.
These bounds supplement, never replace, the actual multiseed G1 IK search.
"""
import numpy as np
import mujoco


def outer_enclosures(g1):
    model=g1.model
    result=[]
    for side in ('left','right'):
        root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,side+'_shoulder_pitch_link')
        cur=g1.wrist_ids[side];chain=[]
        while cur!=root:
            if cur<=0:raise ValueError('wrist is not a shoulder descendant')
            chain.append(int(cur));cur=int(model.body_parentid[cur])
        chain=chain[::-1];joint=int(model.body_jntadr[root]);axis=model.jnt_axis[joint]
        assert np.linalg.norm(model.jnt_pos[joint])<1e-15
        for body in [root,*chain]:
            for j in range(model.body_jntadr[body],model.body_jntadr[body]+model.body_jntnum[body]):
                assert model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
                assert np.linalg.norm(model.jnt_pos[j])<1e-15
        vectors=np.array([model.body_pos[b] for b in chain]);lengths=np.linalg.norm(vectors,axis=1)
        origin=g1.data.xpos[root].copy();rotation=g1.data.xmat[root].reshape(3,3)
        parallel=axis*np.dot(vectors[0],axis)
        result.append({'side':side,'body_chain':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in chain],
            'link_translation_lengths_m':lengths.tolist(),
            'balls':[{'center_m':origin.tolist(),'radius_m':float(lengths.sum()),'derivation':'triangle inequality over every chain translation'},
                     {'center_m':(origin+rotation@parallel).tolist(),'radius_m':float(np.linalg.norm(vectors[0]-parallel)+lengths[1:].sum()),
                      'derivation':'first hinge axial translation is invariant; its transverse orbit plus remaining translation lengths encloses every wrist position'}]})
    return result


def residual_lower_bounds(target,enclosures):
    return np.array([max(0.,*(np.linalg.norm(np.asarray(point)-b['center_m'])-b['radius_m'] for b in arm['balls'])) for point,arm in zip(target,enclosures)])
