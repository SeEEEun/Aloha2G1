"""Common bounded preparation of an idle hand before object-carry motion.

The source contact goal stays fixed. Every preparation and carry edge uses the
caller's complete contact-mode geometry checker; this is offline connection,
not a runtime arm correction or a stored successful posture.
"""
import numpy as np
import mujoco
from .morphology_repair import world_wrist
from .planner import realize_phase_goals


def connect(g, goal, previous, config, validator, seeds, passive_side, object_extent):
    if passive_side in goal['active_hands']:
        raise ValueError('Preparation hand must be passive in the supplied carry goal')
    g.assign(previous);wrist=world_wrist(g,passive_side)
    shoulder=mujoco.mj_name2id(g.model,mujoco.mjtObj.mjOBJ_BODY,passive_side+'_shoulder_pitch_link')
    if shoulder<0:raise ValueError('Named passive shoulder missing')
    center=np.asarray(goal['object_pose'])[:3,3]
    directions=[g.model_to_world_position(g.data.xpos[shoulder])-wrist[:3,3],
                wrist[:3,3]-center,np.array([0.,0.,1.])]
    directions=[v/np.linalg.norm(v) for v in directions if np.linalg.norm(v)>1e-8]
    attempts=[]
    for fraction in (.5,1.):
        for direction in directions:
            target=wrist.copy();target[:3,3]+=fraction*object_extent*direction
            parking=dict(goal,name=goal['name']+'_CONNECT',active_hands=[passive_side],
                wrist_pose_world={passive_side:target},hard_pose_constraint=True,
                orientation_region='SO3',connection_preposes=[])
            parking.pop('preferred_q',None)
            result=realize_phase_goals(g,[parking,goal],previous,config,validator,seeds)
            q=result.pop('q')
            valid=len(q)==3 and all(p['admissible'] for p in result['phases'])
            attempts.append(dict(fraction=fraction,direction=direction,parking_goal=parking,
                q=q,result=result,valid=valid))
            if not valid:continue
            knots=[np.asarray(previous).copy()]
            for index,phase in enumerate(result['phases']):
                subpath=phase.get('connecting_q')
                knots.extend(np.asarray(subpath)[1:] if subpath is not None else [q[index+1]])
            merged=dict(result['phases'][-1],phase=goal['name'],connecting_q=np.asarray(knots),
                passive_hand_preparation=dict(side=passive_side,attempts=attempts,
                    object_extent_m=object_extent,
                    provenance='Half/full modeled object short extent toward the named shoulder, away from the source-conditioned carry object goal, or upward. No source ID, outcome, world-pose constant, or contact permission change.'))
            quality=[p.get('selected_path_quality') for p in result['phases']]
            if all(quality):
                merged['selected_path_quality']=dict(quality[-1],score=float(np.mean([v['score'] for v in quality])),
                    joint_path_length_rad=sum(v['joint_path_length_rad'] for v in quality),
                    cartesian_wrist_path_length_m=sum(v['cartesian_wrist_path_length_m'] for v in quality),
                    preparation_included=True,subpath_quality=quality)
            return dict(result,q=np.asarray([previous,q[-1]]),phases=[merged]),attempts
    return None,attempts
