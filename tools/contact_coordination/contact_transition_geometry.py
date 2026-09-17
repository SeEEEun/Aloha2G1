"""Nominal receiver closing sweep from the existing common hand controller.

This is offline predicted geometry. It neither executes commands nor claims
that commanded fingers equal measured fingers during dynamic contact.
"""
from functools import lru_cache
import numpy as np
from .io import ROOT,read


def receiver_closing_sequence():
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    from .scientific_cache import content_hash
    paths=[PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER,
        ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json']
    paths += [ROOT/'tools/contact_coordination/execution_timing.py',ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json']
    return _receiver_sequence(tuple(content_hash(p) for p in paths)).copy()


@lru_cache(maxsize=8)
def _receiver_sequence(dependency_hashes):
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    from tools.direct_physical_execution_layer import DirectPhysicalDex3ExecutionLayer,authoritative_joint_limits
    from .execution_timing import common_primitive
    primitive=common_primitive()
    lower,upper,names=authoritative_joint_limits(read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json'))
    frames=primitive.preshape_frames+primitive.close_frames
    dummy=np.zeros((frames,28))
    controller=DirectPhysicalDex3ExecutionLayer(primitive,np.asarray(['HANDOFF_INTENT']*frames),dummy,dummy,
        'ACT-A40',lower[:14],upper[:14],lower[14:],upper[14:])
    # Relative primitive time only; no episode clock or source frame is used.
    controller.right_trigger=0;controller.right_start_q=controller.right_open.copy()
    return np.asarray([controller._right_target(i) for i in range(frames)])


def check_receiver_closing(checker,q,left_fingers,object_pose,coarse=False):
    bad=[];sequence=receiver_closing_sequence()
    sweep=[sequence[0]]
    for a,b in zip(sequence[:-1],sequence[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/.005)))
        sweep.extend(a+u*(b-a) for u in np.linspace(0,1,n+1)[1:])
    if coarse:sweep=[sweep[i] for i in np.linspace(0,len(sweep)-1,9).astype(int)]
    for sample,right_fingers in enumerate(sweep):
        command=np.r_[q,np.asarray(left_fingers)[:7],right_fingers]
        bad.extend(dict(receiver_closing_sample=sample,**hit) for hit in checker.query(command,object_pose,('left','right'),object_environment=True)
                   if not hit['allowed_contact'])
    return dict(valid=not bad,samples=len(sweep),forbidden_count=len(bad),forbidden_contacts=bad,
        connection_method='COARSE_CONTACT_REJECTION_ONLY' if coarse else 'CONSTRAINED_LOCAL_CONTACT_MOTION',arm_motion_rad=0.,
        maximum_finger_edge_increment_rad=None if coarse else .005,coarse_pass_never_certifies_motion=coarse)


def check_giver_closing(checker,q,open_fingers,object_pose):
    """Static-arm acquisition with the exact common open/preshape/close chart."""
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    primitive=Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER))
    knots=[np.asarray(open_fingers)[:7],primitive.left_preshape,primitive.left_full_close]
    bad=[];count=0
    for a,b in zip(knots[:-1],knots[1:]):
        for u in np.linspace(0.,1.,max(1,int(np.ceil(np.max(np.abs(b-a))/.005)))+1):
            command=np.r_[q,a+u*(b-a),np.asarray(open_fingers)[7:]]
            bad.extend(dict(giver_closing_sample=count,**h) for h in checker.query(command,object_pose,('left',)) if not h['allowed_contact'])
            count+=1
    return dict(valid=not bad,samples=count,forbidden_contacts=bad,
        connection_method='CONSTRAINED_LOCAL_CONTACT_MOTION',arm_motion_rad=0.,maximum_finger_edge_increment_rad=.005,
        object_table_support_allowed=True)
