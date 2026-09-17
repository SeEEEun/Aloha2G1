"""Finite deterministic charts of source-conditioned contact/phase goals.

Only source relations, object geometry and named robot morphology are inputs.
The region sampler has no physics-result input and no source-ID branches.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .morphology_repair import wrist_target
from .source_phase import mean_pose


def perturbations(radius):
    # Local long-axis grasp translation, approach depth and bounded yaw. The
    # nominal object-relative grasp side and left/right roles are preserved.
    return [(np.zeros(3),0.,0.),(np.array([radius,0.,0.]),0.,.25),
            (np.array([-radius,0.,0.]),0.,-.25),
            (np.array([0.,0.,radius/2]),.025,.5),
            (np.array([0.,0.,-radius/2]),-.025,-.5)]


def acquisition_bank(phase, calibration, dimensions, practical=None):
    from .practical_parameters import DEFAULT
    practical=DEFAULT if practical is None else practical
    if phase['hand_roles']!={'giver':'left','receiver':'right'}:raise ValueError('Unsupported source event roles')
    source=np.asarray(phase['source_functional_tool_object_relations']['left'])
    coordinate=np.linalg.inv(source)[:3,3]/(np.asarray(dimensions)/2)
    radius=min(.003,float(np.min(dimensions))*.04)*(practical['grasp_translation_radius_m']/.003)
    # Smooth bounded prior shift from the source contact location; no clipping
    # that can collapse the five distinct samples to the same target.
    prior=radius*.25*coordinate/np.sqrt(1.+coordinate@coordinate)
    obj=np.asarray(phase['initial_object_pose_world']);rows=[]
    for index,(translation,yaw,depth) in enumerate(perturbations(radius)):
        yaw*=practical['grasp_yaw_max_rad']/.025
        delta=np.eye(4);delta[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
        delta[:3,3]=prior+translation
        contacts=copy.deepcopy(calibration)
        # Rigid object-frame chart action applied to the entire left grasp and
        # loaded relation family. Measured-to-command calibration stays explicit.
        for contact in contacts['contacts'].values():
            if contact['side']!='left':continue
            contact['T_HO']=np.asarray(contact['T_HO'])@np.linalg.inv(delta)
            if 'measured_T_HO' in contact:contact['measured_T_HO']=np.asarray(contact['measured_T_HO'])@np.linalg.inv(delta)
        targets={name:wrist_target(obj,contacts['contacts'][key]) for name,key in
                 [('PREGRASP','pregrasp'),('LEFT_ACQUISITION','acquisition_intent')]}
        # Source approach conditions the final short ingress inside the same
        # bounded contact chart. Preserve the calibrated grasp-side normal;
        # only its lateral pregrasp offset can vary (at most the existing chart
        # radius). A source approach from behind the G1 grasp face therefore
        # cannot reverse the contact side or force an object-crossing ingress.
        source_axis=np.asarray(phase['approach_axis_world'],float)
        source_axis=source_axis/max(np.linalg.norm(source_axis),1e-12)
        nominal=targets['LEFT_ACQUISITION'][:3,3]-targets['PREGRASP'][:3,3]
        normal=nominal/max(np.linalg.norm(nominal),1e-12)
        lateral=source_axis-(source_axis@normal)*normal
        from .runtime_hulls import protected_object_separation
        setback=2*protected_object_separation()
        targets['PREGRASP'][:3,3]-=radius*lateral+setback*normal
        ingress=targets['LEFT_ACQUISITION'][:3,3]-targets['PREGRASP'][:3,3]
        targets['APPROACH_CLEARANCE']=targets['PREGRASP'].copy()
        targets['APPROACH_CLEARANCE'][2,3]+=practical['approach_clearance_m']+depth*radius
        targets['LIFT']=targets['LEFT_ACQUISITION'].copy();targets['LIFT'][2,3]+=.063+depth*radius
        overrides={name:dict(wrist_pose_world={'left':target}) for name,target in targets.items()}
        rows.append(dict(candidate_id=phase['source_id']+':GRASP:'+str(index),source_id=phase['source_id'],
            phase='LEFT_GRASP',source_relation=source,local_perturbation=dict(translation_m=prior+translation,yaw_rad=yaw,approach_depth_m=depth*radius),
            mode_seed='object_local_chart_'+str(index),task_space_target=targets['LEFT_ACQUISITION'],
            targets=targets,goal_overrides=overrides,calibration=contacts,chart_transform=delta,
            acquisition_direction=dict(source_axis_world=source_axis,realized_prior_world=ingress/np.linalg.norm(ingress),
                pregrasp_lateral_offset_world_m=-radius*lateral,maximum_lateral_freedom_m=radius,
                pregrasp_normal_setback_m=setback,minimum_protected_separation_m=protected_object_separation(),
                rule='Source approach projected into bounded grasp-side-preserving G1 pregrasp chart; exact protected state and edge checks remain mandatory'),
            IK_result=None,geometry_result=None,planner_result=None,
            score=dict(source_deviation=float(np.sum((prior+translation)**2)/radius**2+yaw**2))))
    return rows


def handoff_pair_ranking(left,right,coupled,position_scale=.003,rotation_scale=.05):
    """B ranks separable unary sums; C explicitly adds shared-object error.

    Common path/contact validity is a subsequent filter, never used to modify
    these representation-level unary rankings or to silently recouple B.
    """
    rows=[]
    for li,l in enumerate(left):
        for ri,r in enumerate(right):
            a=np.asarray(l['object_pose']);b=np.asarray(r['object_pose'])
            dp=float(np.linalg.norm(a[:3,3]-b[:3,3]))
            dr=float(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())
            cross=(dp/position_scale)**2+(dr/rotation_scale)**2
            unary=float(l['unary_score']+r['unary_score'])
            rows.append(dict(left=li,right=ri,left_candidate_id=l['candidate_id'],right_candidate_id=r['candidate_id'],
                unary_score=unary,cross_hand_score=cross,coupling_weight=1. if coupled else 0.,
                score=unary+(cross if coupled else 0.),position_disagreement_m=dp,rotation_disagreement_rad=dr,
                shared_object_pose=mean_pose([a,b]) if coupled else None,
                representation='C_COUPLED_SHARED_OBJECT' if coupled else 'B_INDEPENDENT_UNARY_SELECTION'))
    return sorted(rows,key=lambda row:(row['score'],row['left_candidate_id'],row['right_candidate_id']))
