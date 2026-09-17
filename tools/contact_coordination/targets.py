"""Phase goals and a single explicit cross-hand selection switch.

No source IDs, episode indices, robot configurations or successful task poses
are consulted. All poses use the same world registration.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .source_phase import mean_pose


def se3_error(first, second):
    a, b = np.asarray(first), np.asarray(second)
    return np.linalg.norm(a[:3, 3]-b[:3, 3]), Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()


def selection_costs(candidates, unary, object_predictions, config, enable_coupling):
    """All N² combinations share unary scores and candidate feasibility.

    The caller supplies collision/kinematic feasibility; no method rescue is
    available here. Independent selection minimizes the identical unary sum.
    """
    rows = []
    for i, left in enumerate(candidates['left']):
        for j, right in enumerate(candidates['right']):
            differences = [se3_error(a, b) for a, b in zip(object_predictions['left'][i], object_predictions['right'][j])]
            p, r = np.asarray(differences).T
            cross = float(np.mean((p/config['position_scale_m'])**2 + (r/config['orientation_scale_rad'])**2))
            base = float(unary['left'][i] + unary['right'][j])
            rows.append(dict(left=i, right=j, unary_cost=base, cross_hand_cost=cross,
                             total_cost=base + (config['coupling_weight']*cross if enable_coupling else 0.),
                             max_shared_object_position_error_m=float(p.max()),
                             max_shared_object_orientation_error_rad=float(r.max())))
    return sorted(rows, key=lambda x: (x['total_cost'], x['left'], x['right']))


def make_goals(phase, priors, config, representation='INTERACTION_OURS', enable_coupling=True):
    if representation not in ('WRIST_REFERENCE', 'INTERACTION_OURS'):
        raise ValueError(representation)
    times = np.asarray(priors['source_timestamp'])
    events = {k: v['time_s'] for k, v in phase['events'].items()}
    wrists = {s: np.asarray(priors[s+'_wrist_world']) for s in ('left', 'right')}
    relation = {s: np.asarray(phase['registered_wrist_object_relations'][s]) for s in wrists}
    giver, receiver = phase['hand_roles']['giver'], phase['hand_roles']['receiver']
    if {giver, receiver} != {'left', 'right'}:
        raise ValueError('roles must name two distinct hands')
    inferred = wrists[giver] @ relation[giver]
    at = lambda event: int(np.argmin(abs(times-events[event])))
    # Three overlap goals constrain a segment, not an isolated handoff frame.
    overlap = np.flatnonzero((times >= events['RIGHT_ACQUIRE_SOURCE']) & (times <= events['LEFT_RELEASE_BEGIN']))
    anchors = overlap[np.linspace(0, len(overlap)-1, 3).round().astype(int)]
    names_frames = [('pregrasp', at('LEFT_CLOSE_BEGIN')), ('acquisition', at('LEFT_GRASP_SOURCE')),
                    ('lift_clearance', at('LEFT_TRANSPORT_BEGIN')),
                    *[(f'handoff_{k}', int(f)) for k, f in enumerate(anchors)],
                    ('right_transport', at('RIGHT_TRANSPORT_BEGIN')), ('placement', at('FINAL_RELEASE_BEGIN'))]
    targets = []
    selection = None
    if representation == 'INTERACTION_OURS':
        # Candidate bank consists only of source-supported rigid object poses.
        # Common offsets span observed overlap variation, with no added workspace.
        center = mean_pose(inferred[anchors])
        offsets = [inferred[f] @ np.linalg.inv(center) for f in anchors]
        candidates = {s: [] for s in wrists}; predictions = {s: [] for s in wrists}; unary = {s: [] for s in wrists}
        for side in wrists:
            for offset in offsets:
                objects = offset @ inferred[anchors]
                goals = objects @ np.linalg.inv(relation[side])
                cost = sum((se3_error(a, b)[0]/config['position_scale_m'])**2 +
                           (se3_error(a, b)[1]/config['orientation_scale_rad'])**2
                           for a, b in zip(goals, wrists[side][anchors])) / len(anchors)
                candidates[side].append(goals); predictions[side].append(objects); unary[side].append(cost)
        ranking = selection_costs(candidates, unary, predictions, config, enable_coupling)
        selection = dict(enable_coupling=enable_coupling, ranking=ranking,
                         selected=ranking[0], candidate_object_offsets=offsets,
                         noncoupling_settings=copy.deepcopy(config),
                         candidate_source_frames=anchors, simultaneous_ik_checked=False)
    for name, frame in names_frames:
        goal = {s: wrists[s][frame].copy() for s in wrists}
        object_pose = inferred[frame].copy()
        if representation == 'INTERACTION_OURS':
            if name in ('pregrasp', 'acquisition'):
                object_pose = np.asarray(phase['initial_object_pose_world'])
                goal[giver] = object_pose @ np.linalg.inv(relation[giver])
                if name == 'pregrasp':
                    axis = phase['approach_axis_world']
                    if axis is None:
                        raise ValueError('SOURCE_EVIDENCE_MISSING: approach axis')
                    goal[giver][:3, 3] -= config['pregrasp_retreat_m'] * np.asarray(axis)
            elif name == 'lift_clearance':
                goal[giver] = object_pose @ np.linalg.inv(relation[giver])
            elif name.startswith('handoff_'):
                k = int(name.rsplit('_', 1)[1])
                for s in wrists:
                    goal[s] = candidates[s][selection['selected'][s]][k]
                object_pose = goal[giver] @ relation[giver]
            else:
                object_pose = wrists[receiver][frame] @ relation[receiver]
                goal[receiver] = object_pose @ np.linalg.inv(relation[receiver])
        active = [giver] if name in ('pregrasp', 'acquisition', 'lift_clearance') else [giver, receiver] if name.startswith('handoff_') else [receiver]
        targets.append(dict(name=name, source_time_s=float(times[frame]), source_frame=int(frame),
                            wrist_pose_world=goal, active_hands=active,
                            predicted_object_pose_world=object_pose,
                            reference_frame='shared_registered_world',
                            position_tolerance_m=config['phase_position_tolerance_m'],
                            orientation_tolerance_rad=config['phase_orientation_tolerance_rad'],
                            contact_relation_status=phase['relation_status'],
                            transition='measured contact candidate before lift; measured right retention after giver release',
                            source_input='phase events + registered object/wrist relations' if representation == 'INTERACTION_OURS' else 'calibrated registered wrist/TCP prior'))
    return dict(schema='hybrid_phase_targets_v1', source_id=phase['source_id'], hand_roles=phase['hand_roles'],
                representation=representation, enable_coupling=enable_coupling if representation=='INTERACTION_OURS' else None,
                phase_goals=targets, contact_relations=relation, candidate_selection=selection,
                dense_source_motion_role='spatial prior and diagnostic; no absolute source-time equality',
                contact_patch_certification=phase['supported_grasp_patches'],
                object_attachment_allowed=False)
