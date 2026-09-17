"""Predeclared parameter-only TRAIN40 sweep; no outcome-dependent recipes."""
from pathlib import Path
from .io import read

DEFAULT = dict(approach_clearance_m=.085, grasp_translation_radius_m=.003,
               grasp_yaw_max_rad=.025, incidental_penetration_m=0.,
               incidental_displacement_m=0.)
BOUNDS = dict(approach_clearance_m=(.015,.085),grasp_translation_radius_m=(.003,.006),
              grasp_yaw_max_rad=(.025,.075),incidental_penetration_m=(0.,.002),
              incidental_displacement_m=(0.,.008))
RECIPES = {
    'CURRENT_DEFAULT': dict(DEFAULT,acquisition_posture_multiplier=1.),
    'LOWER_CLEARANCE': dict(DEFAULT,approach_clearance_m=.055,
        incidental_penetration_m=.0005,incidental_displacement_m=.003,acquisition_posture_multiplier=1.),
    'COMPACT_APPROACH': dict(DEFAULT,approach_clearance_m=.030,grasp_translation_radius_m=.0045,
        grasp_yaw_max_rad=.050,incidental_penetration_m=.001,incidental_displacement_m=.005,
        acquisition_posture_multiplier=.5),
    'NEAR_APPROACH': dict(DEFAULT,approach_clearance_m=.015,grasp_translation_radius_m=.006,
        grasp_yaw_max_rad=.075,incidental_penetration_m=.002,incidental_displacement_m=.008,
        acquisition_posture_multiplier=.25),
}


def parameters(out=None):
    path=Path(out)/'PRACTICAL_PARAMETERS.json' if out is not None else None
    values=read(path)['values'] if path is not None and path.exists() else DEFAULT
    if set(values)!=set(DEFAULT):raise ValueError('Unsupported practical calibration dimension')
    for key,value in values.items():
        low,high=BOUNDS[key]
        if not low<=float(value)<=high:raise ValueError('Practical parameter out of bounds: '+key)
    return {key:float(value) for key,value in values.items()}
