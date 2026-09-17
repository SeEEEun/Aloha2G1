"""Source closing-line alternatives around the calibrated Dex3 grasp family.

All alternatives preserve the supported initial object pose. Changing a grasp
family changes every giver object-in-hand transform, not only the first wrist.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def families(phase, calibration):
    side=phase['hand_roles']['giver']
    source=np.linalg.inv(np.asarray(phase['source_functional_tool_object_relations'][side]))
    nominal=np.linalg.inv(np.asarray(calibration['contacts']['acquisition_intent']['T_HO']))
    object_world=np.asarray(phase['initial_object_pose_world'])
    up=object_world[:3,:3].T@np.array([0.,0.,1.])
    a=source[:3,1];b=nominal[:3,0]
    a=a-up*(up@a);b=b-up*(up@b)
    if min(np.linalg.norm(a),np.linalg.norm(b))<1e-8:
        raise ValueError('SOURCE_EVIDENCE_MISSING: acquisition closing line in support plane')
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    angle=float(np.arctan2(up@np.cross(b,a),b@a))
    angle=(angle+np.pi/2)%np.pi-np.pi/2
    values=[]
    for kind,rotation in [('CALIBRATED_CONTACT_PRIOR',0.),('SOURCE_CLOSING_LINE',angle),
                          ('SOURCE_CLOSING_LINE_ANTIPODAL',angle+np.pi)]:
        d=np.eye(4);d[:3,:3]=Rotation.from_rotvec(up*rotation).as_matrix()
        if any(np.allclose(d,v['D_object_contact'],atol=1e-12,rtol=0) for v in values):continue
        values.append(dict(kind=kind,D_object_contact=d,rotation_rad=rotation,
            source_id=phase['source_id'],source_field='source_functional_tool_object_relations.'+side,
            source_closing_line_object=a,target_opposition_line_object=d[:3,:3]@b,
            source_axis_error_rad=float(np.arccos(np.clip(abs(a@(d[:3,:3]@b)),0.,1.))),
            region_provenance='Existing calibrated contact as capability prior, plus two signs of the observed/inferred ALOHA jaw-closing line mapped to Dex3 opposition. Rotation only about support gravity; no initial object movement or full tool-frame equality.',
            qualification='Candidate only: full geometry, connections, and dynamic contact verification required',
            selection_rule='Deterministic calibration-prior first, then source-conditioned closing-line alternatives; no physical outcome or source-ID branch'))
    return values


def apply(calibration, family):
    result=copy.deepcopy(calibration);d=np.asarray(family['D_object_contact']);inverse=np.linalg.inv(d)
    side=result['contacts']['acquisition_intent']['side']
    for contact in result['contacts'].values():
        if contact.get('side')!=side:continue
        for field in ('T_HO','measured_T_HO'):
            if field in contact:contact[field]=np.asarray(contact[field])@inverse
    result['source_acquisition_family']=family
    return result
