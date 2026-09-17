"""Retiming of the unchanged Dex3 contact paths using common joint limits."""
from dataclasses import replace
import numpy as np
from .io import ROOT,read


def retime_primitive(primitive):
    joints=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')['joints'][14:]
    velocity=np.array([j['max_velocity_rad_s'] for j in joints]);acceleration=np.array([j['max_acceleration_rad_s2'] for j in joints])
    changes={}
    for field,start,end in [('preshape_frames','open','preshape'),('close_frames','preshape','full_close'),('release_frames','full_close','open')]:
        delta=abs(np.r_[getattr(primitive,'left_'+end)-getattr(primitive,'left_'+start),
                        getattr(primitive,'right_'+end)-getattr(primitive,'right_'+start)])
        seconds=max(float(np.max(1.875*delta/velocity)),float(np.max(np.sqrt((10/np.sqrt(3))*delta/acceleration))))
        changes[field]=max(getattr(primitive,field),int(np.ceil(seconds*30))+1)
    return replace(primitive,**changes)


def common_primitive():
    from tools.common_execution_layer import Dex3Primitive,read_json
    from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    return retime_primitive(Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER)))
