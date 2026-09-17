"""Causal retiming of feedback-generated finger contact commands at 30 Hz.

Contact debounce is an evidence counter, not a motion speed. Keep already
admissible command samples and retime discontinuities using the same limits as
the geometric trajectory. No arm command or contact decision is changed.
"""
import numpy as np
from .io import ROOT,read


def contact_limits():
    config=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')
    rows=config['joints'][14:]
    return (float(config['control_period_s']),
        np.array([r['max_velocity_rad_s'] for r in rows]),
        np.array([r['max_acceleration_rad_s2'] for r in rows]))


def stopping_velocity(distance,acceleration,dt):
    """Largest next discrete velocity that can stop within distance.

    Include the next sample's travel and all subsequent deceleration samples.
    This is the inverse of the finite arithmetic stopping-distance sum.
    """
    distance=np.maximum(0.,distance)
    n=np.maximum(1.,np.ceil((np.sqrt(1.+8.*distance/(acceleration*dt*dt))-1.)/2.))
    return (distance/dt+acceleration*dt*n*(n-1.)/2.)/n


class ContactCommandRetimer:
    def __init__(self,velocity,acceleration,lower,upper,dt):
        self.vmax=np.asarray(velocity);self.amax=np.asarray(acceleration)
        self.lower=np.asarray(lower);self.upper=np.asarray(upper);self.dt=float(dt)
        self.q=None;self.v=np.zeros_like(self.vmax);self.recovering=np.zeros_like(self.vmax,dtype=bool)
        self.changed_frames=0;self.maximum_velocity_ratio=0.;self.maximum_acceleration_ratio=0.

    def step(self,target):
        target=np.clip(np.asarray(target,dtype=float),self.lower,self.upper)
        if not np.isfinite(target).all():raise ValueError('NONFINITE_CONTACT_TARGET')
        if self.q is None:self.q=target.copy();return target.copy()
        dt=self.dt;dv=self.amax*dt
        lower=np.maximum.reduce([-self.vmax,self.v-dv,-stopping_velocity(self.q-self.lower,self.amax,dt)])
        upper=np.minimum.reduce([self.vmax,self.v+dv,stopping_velocity(self.upper-self.q,self.amax,dt)])
        if np.any(lower>upper+1e-10):raise RuntimeError('RETIMING_FAIL: contact command outside viable braking state')
        requested=(target-self.q)/dt
        self.recovering|=(requested<lower-1e-10)|(requested>upper+1e-10)
        stopping=np.sign(requested)*stopping_velocity(abs(target-self.q),self.amax,dt)
        desired=np.where(self.recovering,stopping,requested)
        velocity=np.clip(desired,lower,upper)
        q=np.clip(self.q+velocity*dt,self.lower,self.upper)
        velocity=(q-self.q)/dt
        changed=bool(np.max(abs(q-target))>1e-10)
        self.changed_frames+=int(changed)
        self.maximum_velocity_ratio=max(self.maximum_velocity_ratio,float(np.max(abs(velocity)/self.vmax)))
        self.maximum_acceleration_ratio=max(self.maximum_acceleration_ratio,float(np.max(abs(velocity-self.v)/dv)))
        self.recovering&=~((abs(q-target)<1e-10)&(abs(velocity)<=dv+1e-10))
        self.q=q;self.v=velocity
        return q.copy()

    def summary(self):
        return dict(changed_frames=self.changed_frames,maximum_velocity_ratio=self.maximum_velocity_ratio,
            maximum_acceleration_ratio=self.maximum_acceleration_ratio,
            motion_type='CONSTRAINED_LOCAL_CONTACT_MOTION',arm_targets_modified=False,
            contact_debounce_frames_changed=False,contact_target_amplitudes_changed=False,
            validation='Nominal contact paths checked before physics; actual adaptive contact commands and measured substeps audited separately')


def audit_issued_commands(trace):
    """Check the commands actually issued, including contact-latch boundaries."""
    indices=np.r_[np.flatnonzero(np.diff(trace['control_frame'])!=0),len(trace['control_frame'])-1]
    config=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json')
    rows=config['joints'];dt=float(config['control_period_s'])
    q=trace['EXECUTED_COMMAND'][indices]
    velocity=abs(np.diff(q,axis=0)/dt);acceleration=abs(np.diff(q,n=2,axis=0)/(dt*dt))
    v=np.array([r['max_velocity_rad_s'] for r in rows]);a=np.array([r['max_acceleration_rad_s2'] for r in rows])
    vr=velocity/v;ar=acceleration/a
    return dict(status='PASS' if np.max(vr)<=1.+1e-6 and np.max(ar)<=1.+1e-6 else 'RETIMING_FAIL',
        frames=len(q),control_period_s=dt,joint_names=[r['joint_name'] for r in rows],
        maximum_velocity_ratio_per_joint=np.max(vr,axis=0),maximum_acceleration_ratio_per_joint=np.max(ar,axis=0),
        velocity_violations=np.argwhere(vr>1.+1e-6),acceleration_violations=np.argwhere(ar>1.+1e-6),
        source='ACTUAL_ISSUED_PHYSICS_COMMANDS',checks_include_feedback_preload_and_release=True)
