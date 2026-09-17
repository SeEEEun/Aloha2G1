"""Source receiving intent in a bounded chart of calibrated Dex3 contacts.

Source TCP X is approach, Y is jaw closing. Dex3 H X is thumb/opposition
closing. The transforms are therefore NOT equated. Source object-relative
grasp-center and unsigned closing-line direction condition a local contact
region; the measured target primitive supplies its origin and morphology.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation


def build_region(phase, contact, dimensions, separation, gravity_up):
    if phase['hand_roles']['receiver'] != contact['side']:
        raise ValueError('Receiving relation/calibration side mismatch')
    side=phase['hand_roles']['receiver']
    source=np.asarray(phase['source_functional_tool_object_relations'][side],float)
    if source.shape!=(4,4) or not np.isfinite(source).all():
        raise ValueError('SOURCE_EVIDENCE_MISSING: finite source functional receiving relation')
    source_hand=np.linalg.inv(source)
    nominal=np.linalg.inv(np.asarray(contact['T_HO'],float))
    half=np.asarray(dimensions,float)/2
    radius=float(np.linalg.norm(separation))
    if radius<=0 or np.any(half<=0):raise ValueError('Measured target contact-region extent missing')
    # A normalized source grasp location, not source wrist XYZ. This smooth
    # unit-ball chart has no source ID, golden anchor, or outcome-dependent fit.
    # The available displacement is the existing morphology clearance span.
    coordinate=source_hand[:3,3]/half
    patch=coordinate/np.sqrt(1.+float(coordinate@coordinate))
    delta=radius*patch
    up=np.asarray(gravity_up,float);up=up/np.linalg.norm(up)
    source_axis=source_hand[:3,1]  # ALOHA named jaw-closing axis
    target_axis=nominal[:3,0]      # Dex3 thumb/opposition axis
    a=source_axis-up*(up@source_axis);b=target_axis-up*(up@target_axis)
    if min(np.linalg.norm(a),np.linalg.norm(b))<1e-8:
        raise ValueError('SOURCE_EVIDENCE_MISSING: gravity-plane receiving closing line')
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    # Jaw opposition is a line: flipping either axis cannot change the goal.
    signed_sine=float(up@np.cross(b,a));cosine=float(b@a)
    chord_angle=2*np.arcsin(min(1.,radius/float(np.min(dimensions))))
    yaw=chord_angle*signed_sine*cosine
    selected=nominal.copy()
    selected[:3,:3]=Rotation.from_rotvec(up*yaw).as_matrix()@nominal[:3,:3]
    selected[:3,3]+=delta
    return dict(schema='source_receiving_contact_chart_v1',source_id=phase['source_id'],
        source_field='source_functional_tool_object_relations.'+side,
        source_T_TCP_O=source,source_T_O_TCP=source_hand,
        source_status=phase['relation_status'][side],
        source_grasp_coordinate_in_object_half_extents=coordinate,
        source_closing_axis_object=source_axis,source_approach_axis_object=source_hand[:3,0],
        nominal_target_T_O_H=nominal,target_T_O_H=selected,
        target_T_H_O=np.linalg.inv(selected),translation_object_m=delta,
        gravity_axis_object=up,closing_line_rotation_rad=yaw,
        translation_radius_m=radius,closing_line_rotation_bound_rad=chord_angle/2,
        source_to_target_position_error_m=float(np.linalg.norm(source_hand[:3,3]-selected[:3,3])),
        source_to_target_closing_line_error_rad=float(np.arccos(np.clip(abs(source_axis@selected[:3,0]),0.,1.))),
        bounds_provenance='Existing calibrated palm/object separation span and modeled proxy short extent. No DEV outcomes; no enlarged region to admit a failure.',
        realization_rule='Object-normalized source grasp location maps continuously into the target primitive local displacement ball. Unsigned source jaw closing maps to Dex3 opposition about the calibrated gravity axis. Preserve the calibrated palm/gravity realization; retain raw source deviations explicitly.',
        limitations='A local morphology prior, not an exact reproduction of source TCP pose or a certified source contact patch. Source roll/pitch and dynamic object/contact evidence remain inferred/unknown. Full geometry and physical verification remain mandatory.',
        source_left_relation_modified=False,golden_world_or_q_used=False)


def realize_contact(contact, region):
    result=copy.deepcopy(contact)
    result['T_HO']=np.asarray(region['target_T_H_O'])
    result['source_receiving_region']=region
    return result
