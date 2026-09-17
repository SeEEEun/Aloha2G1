"""Four bounded multipliers of already used converter quantities.

No robot frame, scene, physical tolerance or per-instance budget is a knob.
"""
from .io import read

DEFAULT = dict(acquisition_posture_multiplier=1., connection_posture_multiplier=1.,
               handoff_posture_multiplier=1., receiver_spread_fraction=1.)
BOUNDS = dict(acquisition_posture_multiplier=(.25,4.), connection_posture_multiplier=(.25,4.),
              handoff_posture_multiplier=(.1,1.), receiver_spread_fraction=(.5,1.))
RECIPES = {
    'CURRENT': DEFAULT,
    'SOURCE_PRIOR_STRONGER': dict(acquisition_posture_multiplier=.25,connection_posture_multiplier=.25,
                                handoff_posture_multiplier=.1,receiver_spread_fraction=1.),
    'CENTRAL_CONTACT': dict(acquisition_posture_multiplier=1.,connection_posture_multiplier=1.,
                           handoff_posture_multiplier=.5,receiver_spread_fraction=.5),
    'POSTURE_PRESERVING': dict(acquisition_posture_multiplier=4.,connection_posture_multiplier=4.,
                             handoff_posture_multiplier=1.,receiver_spread_fraction=.75),
}


def parameters(out):
    path=out/'CALIBRATION_PARAMETERS.json'
    values=read(path)['values'] if path.exists() else DEFAULT
    if set(values)!=set(DEFAULT):raise ValueError('Unsupported calibration parameter')
    for key,value in values.items():
        lower,upper=BOUNDS[key]
        if not lower<=float(value)<=upper:raise ValueError('Parameter outside predeclared domain: '+key)
    return {k:float(v) for k,v in values.items()}


def scaled_ik_config(out, config, phase):
    if phase not in ('acquisition','connection'):raise ValueError(phase)
    result=dict(config)
    result['joint_prior_scale']*=parameters(out)[phase+'_posture_multiplier']
    return result
