"""Bounded intended-hand pre-acquisition contact, with unchanged scene physics.

Only an explicit practical configuration activates the relaxed policy. Every
other robot/object pair, deep contact and distant grasp patch stay forbidden.
"""
from pathlib import Path
import numpy as np
from .io import read
from .practical_parameters import parameters


def policy(out):
    values=parameters(out);depth=values['incidental_penetration_m']
    if depth==0.:return dict(enabled=False,maximum_depth_m=0.,maximum_displacement_m=0.)
    cal=read(Path(out)/'target_repair/CONTACT_CALIBRATION.json')
    def wrist_in_object(name):
        contact=cal['contacts'][name]
        return np.linalg.inv(np.asarray(contact['T_wrist_H'])@np.asarray(contact['T_HO']))[:3,3]
    # The functional TCP origin can lie inside the object; its displacement
    # from object center is not a grasp-side normal. The calibrated ingress
    # specifies which object face the intended hand approaches instead.
    origin=wrist_in_object('pregrasp')-wrist_in_object('acquisition_intent')
    from .runtime_hulls import object_dimensions
    half=object_dimensions(out)/2
    if np.linalg.norm(origin)<1e-8:raise ValueError('Incidental policy needs an intended grasp-side relation')
    center=origin/np.sqrt(np.sum((origin/half)**2))
    hulls=read(Path(out)/'target_repair/runtime_bin150/RUNTIME_HULLS.json')
    # The query model welds this fixed palm component to the wrist body.
    # Preserve collision-shape identity; never admit the whole wrist body.
    palm_components={r['key']:'left_hand_palm_link' for r in hulls['rows']
        if r['kind']=='robot_convex' and '/left_hand_palm_link/' in r['path']}
    if not palm_components:raise ValueError('Authored intended palm collision component missing')
    return dict(enabled=True,intended_hand='left',maximum_depth_m=depth,
        maximum_displacement_m=values['incidental_displacement_m'],maximum_rotation_rad=.12,
        maximum_linear_speed_m_s=.5,maximum_angular_speed_rad_s=10.,
        patch_center_object_m=center,patch_radius_m=.035,
        patch_definition='Object face approached by calibrated pregrasp-to-acquisition wrist displacement',
        patch_side_normal_object=origin/np.linalg.norm(origin),
        minimum_patch_side_projection_m=.005,
        allowed_body_rule='Left thumb/index/middle fingers or authored left palm component only; no camera/wrist/forearm/arm/torso/opposite hand',
        welded_palm_collision_components=palm_components,
        conservative_outer_bounds=dict(penetration_m=.002,displacement_m=.008),
        collision_geometry_modified=False,controller_modified=False)


def intended_link(name):
    return any(name.startswith('left_hand_'+digit+'_') for digit in ('thumb','index','middle')) or name in ('left_hand_palm_link','left_palm_link','left_hand_link')


def permits(config,link,depth,point_world,object_pose):
    if not config['enabled'] or not intended_link(link):return False
    if not np.isfinite(depth) or depth>config['maximum_depth_m']+1e-9:return False
    point=np.asarray(point_world,float);pose=np.asarray(object_pose,float)
    if point.shape!=(3,) or not np.isfinite(point).all():return False
    local=pose[:3,:3].T@(point-pose[:3,3])
    return bool(np.linalg.norm(local-np.asarray(config['patch_center_object_m']))<=config['patch_radius_m']
        and local@np.asarray(config['patch_side_normal_object'])>=config['minimum_patch_side_projection_m'])


def filter_hits(hits,config,object_pose):
    if not config['enabled']:return hits
    result=[]
    for hit in hits:
        bodies=hit['bodies'];row=dict(hit)
        if 'object' in bodies and len(bodies)==2:
            link=next(b for b in bodies if b!='object')
            component=link
            if link=='left_wrist_yaw_link':
                named=[config.get('welded_palm_collision_components',{}).get(g) for g in hit.get('geoms',[])]
                if 'left_hand_palm_link' in named:component='left_hand_palm_link'
            depth=max(0.,-hit['signed_distance_m']) if 'signed_distance_m' in hit else hit['depth_m']
            if permits(config,component,depth,hit.get('point_world',[np.nan]*3),object_pose):
                row.update(allowed_contact=True,reason='BOUNDED_INTENDED_HAND_INCIDENTAL_CONTACT',
                    intended_contact_component=component,
                    actual_penetration_m=depth,maximum_incidental_penetration_m=config['maximum_depth_m'])
        result.append(row)
    return result


def measured_assessment(config,contacts,trace,acquisition_index,initial):
    """Telemetry only: never changes, truncates or reconstructs a command."""
    from scipy.spatial.transform import Rotation
    count=max(1,acquisition_index);positions=np.asarray(trace['object_position_world_m'])
    rotations=Rotation.from_quat(trace['object_quaternion_xyzw'])
    displacement=float(np.linalg.norm(positions[:count]-np.asarray(initial[:3]),axis=1).max())
    angle=float((Rotation.from_quat(initial[3:]).inv()*rotations[:count]).magnitude().max())
    maximum_linear=float(np.linalg.norm(trace['object_linear_velocity_m_s'][:count],axis=1).max())
    maximum_angular=float(np.linalg.norm(trace['object_angular_velocity_rad_s'][:count],axis=1).max())
    bad=[];times=np.asarray(trace['timestamp_s'])
    for contact in contacts:
        index=int(np.argmin(abs(times-contact['timestamp_s'])))
        pose=np.eye(4);pose[:3,:3]=rotations[index].as_matrix();pose[:3,3]=positions[index]
        depth=max(0.,-contact['separation_m'])
        if not permits(config,contact['robot_link'],depth,contact.get('point_world_m',[np.nan]*3),pose):bad.append(contact)
    motion_valid=(displacement<=config['maximum_displacement_m']+1e-9 and angle<=config['maximum_rotation_rad']
        and maximum_linear<=config['maximum_linear_speed_m_s'] and maximum_angular<=config['maximum_angular_speed_rad_s'])
    # Passive settling with no contact is still measured. Relaxed configurations
    # explicitly bound all pre-acquisition motion, not only detected contact.
    return dict(accepted=not bad and motion_valid,rejected_contacts=bad[:30],rejected_count=len(bad),
        maximum_depth_m=max((max(0.,-c['separation_m']) for c in contacts),default=0.),
        maximum_contact_impulse_ns=max((c['force_n']/240. for c in contacts),default=0.),
        maximum_object_linear_speed_before_acquisition_m_s=maximum_linear,
        maximum_object_angular_speed_before_acquisition_rad_s=maximum_angular,
        displacement_within_policy=displacement<=config['maximum_displacement_m']+1e-9,
        rotation_within_policy=angle<=config['maximum_rotation_rad'],policy=config)
