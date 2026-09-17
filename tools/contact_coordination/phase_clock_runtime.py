"""Common phase release gate; qualified contact and finger rules are unchanged."""
from dataclasses import replace
import numpy as np
from tools.direct_physical_execution_layer import DirectPhysicalDex3ExecutionLayer,authoritative_joint_limits
from tools.direct_physical_execution_isaac_runtime import build_runtime as original_runtime,JOINT_CONTRACT
from tools.common_execution_layer import read_json,_transition,_minimum_jerk
from tools.direct_physical_execution_layer import DIGIT_LOCAL_INDICES


def giver_release_target(start,opened,elapsed,frames,policy):
    """Same qualified OPEN endpoints/rates with an explicit digit order."""
    if policy=='simultaneous':return _transition(start,opened,elapsed,frames)
    if policy not in ('thumb_first','middle_first'):raise ValueError('Unknown common giver release policy')
    result=np.asarray(start).copy()
    first=np.arange(3) if policy=='thumb_first' else np.arange(3,5)
    later=np.setdiff1d(np.arange(7),first)
    result[first]=_transition(np.asarray(start)[first],np.asarray(opened)[first],elapsed,frames)
    result[later]=_transition(np.asarray(start)[later],np.asarray(opened)[later],elapsed-frames,frames)
    return result


def release_geometry_fingers(left_start,right_start,left_open,right_hold,elapsed,frames,policy='simultaneous',base_frames=None):
    """Shared predicted finger/contact transition for the planned release edge."""
    base_frames=int(base_frames or frames)
    progress=elapsed/max(1.,frames-1.) if policy=='simultaneous' else (elapsed-base_frames)/max(1.,base_frames-1.)
    fraction=float(_minimum_jerk(progress))
    left=giver_release_target(left_start,left_open,elapsed,frames if policy=='simultaneous' else base_frames,policy)
    right=(1-fraction)*np.asarray(right_start)+fraction*np.asarray(right_hold)
    return np.r_[left,right],fraction


def release_arm_fraction(elapsed,frames,policy,base_frames):
    """Unload middle while stationary, then withdraw during remaining release."""
    delay=base_frames if policy=='middle_first' else 0
    return float(_minimum_jerk((elapsed-delay)/max(1.,frames-delay)))


def receiver_release_target(start,opened,frame,begin,frames,delayed_digits=(),delayed_begin=None):
    """Open unobstructed digits, then remaining digits after planned clearance."""
    result=_transition(np.asarray(start),np.asarray(opened),frame-begin,frames)
    for digit in delayed_digits:
        indices=DIGIT_LOCAL_INDICES[digit]
        result[indices]=_transition(np.asarray(start)[indices],np.asarray(opened)[indices],
                                   frame-int(delayed_begin),frames)
    return result


class PhaseClockDex3(DirectPhysicalDex3ExecutionLayer):
    def __init__(self,*args,release_not_before_frame,giver_release_policy='simultaneous',coordinated_giver_release=False,coordinated_release_frames=None,receiver_transition_frames=None,final_release_delayed_digits=(),final_release_delayed_begin=None,**kwargs):
        super().__init__(*args,**kwargs)
        from .execution_timing import retime_primitive
        self.primitive=retime_primitive(self.primitive)
        self.release_not_before_frame=int(release_not_before_frame)
        if giver_release_policy not in ('simultaneous','thumb_first','middle_first'):raise ValueError(giver_release_policy)
        self.giver_release_policy=giver_release_policy
        self.base_release_frames=self.primitive.release_frames
        self.coordinated_giver_release=bool(coordinated_giver_release)
        self.coordinated_release_frames=max(self.base_release_frames,int(coordinated_release_frames or self.base_release_frames))
        self.receiver_transition_frames=max(self.primitive.preshape_frames+self.primitive.close_frames,int(receiver_transition_frames or 0))
        self.final_release_delayed_digits=tuple(final_release_delayed_digits)
        if set(self.final_release_delayed_digits)-set(DIGIT_LOCAL_INDICES):raise ValueError('Unknown delayed release digit')
        if 'thumb' in self.final_release_delayed_digits and {'index','middle'}&set(self.final_release_delayed_digits):
            raise ValueError('A delayed opening may not preserve an opposing grasp pair')
        if self.final_release_delayed_digits and final_release_delayed_begin is None:raise ValueError('Delayed opening requires an explicit planned clearance boundary')
        self.final_release_delayed_begin=final_release_delayed_begin
        from .contact_command_retiming import ContactCommandRetimer,contact_limits
        dt,velocity,acceleration=contact_limits()
        self.contact_retimers={side:ContactCommandRetimer(velocity[offset:offset+7],acceleration[offset:offset+7],
            self.dex3_command_lower_rad[offset:offset+7],self.dex3_command_upper_rad[offset:offset+7],dt)
            for side,offset in [('left',0),('right',7)]}

    def _contact_seek(self,side,frame,snapshot,nominal_target,release_active):
        target,events=super()._contact_seek(side,frame,snapshot,nominal_target,release_active)
        return self.contact_retimers[side].step(target),events

    @property
    def giver_release_duration_frames(self):
        if self.coordinated_giver_release:return self.coordinated_release_frames
        return self.base_release_frames*(2 if self.giver_release_policy!='simultaneous' else 1)

    def _left_target(self,frame):
        if self.giver_release_policy!='simultaneous' and self.right_support_frame is not None and self.left_release_frame is not None:
            return giver_release_target(self.left_release_start_q,self.left_open,frame-self.left_release_frame,self.base_release_frames,self.giver_release_policy)
        return super()._left_target(frame)

    def _right_target(self,frame):
        if self.right_release_frame is not None and self.final_release_delayed_digits:
            return receiver_release_target(self.right_release_start_q,self.right_open,frame,
                self.right_release_frame,self.base_release_frames,self.final_release_delayed_digits,self.final_release_delayed_begin)
        return super()._right_target(frame)

    def step(self,frame,snapshot):
        # Same bounded nominal eligibility rule for every representation.
        # The existing source-clock adapter uses this exact temporary primitive
        # replacement to prevent release before the declared release phase.
        original=self.primitive
        overrides={}
        if self.right_trigger is not None or str(self.intent[frame])=='HANDOFF_INTENT':
            n=self.receiver_transition_frames;old_total=original.preshape_frames+original.close_frames
            preshape=int(np.ceil(n*original.preshape_frames/old_total))
            overrides.update(preshape_frames=preshape,close_frames=n-preshape)
        if (self.giver_release_policy!='simultaneous' or self.coordinated_giver_release) and (self.right_trigger is not None or str(self.intent[frame])=='HANDOFF_INTENT') and str(self.intent[frame])!='FINAL_RELEASE_INTENT':
            # The base ownership counter must wait for the entire actual giver
            # opening. Receiver preshape/close and final release keep their rates.
            overrides['release_frames']=self.giver_release_duration_frames
        if frame<self.release_not_before_frame:
            overrides['right_verification_frames']=len(self.safe)+1
        if overrides:self.primitive=replace(original,**overrides)
        previous=snapshot.previous_control_frame_support or {}
        giver_support=bool(previous.get('giver_contact_present',any(float(v)>=original.force_threshold_n for v in snapshot.digit_force_n['left'].values())))
        if giver_support:
            # Open commands do not establish sole receiver ownership. Prevent
            # the inherited counter from issuing ownership while the giver
            # still supports the object, without blocking receiver acquisition.
            self.right_retention_counter=-len(self.safe)-1
            # Keep the first measured ownership event as history. Admission of
            # later transport uses the current retention counter, not this label.
        try:
            result=super().step(frame,snapshot)
            if giver_support:self.right_retention_counter=0
            return result
        finally:self.primitive=original

    def summary(self):
        result=super().summary();result.pop('method',None)
        result.update(controller='COMMON_PHASE_CONTACT_DEX3',release_not_before_frame=self.release_not_before_frame,
            giver_release_policy=self.giver_release_policy,giver_release_duration_frames=self.giver_release_duration_frames,
            coordinated_giver_release=self.coordinated_giver_release,
            final_release_delayed_digits=self.final_release_delayed_digits,final_release_delayed_begin=self.final_release_delayed_begin,
            method_identifier_consumed=False,spatial_arm_feedback=False,grasp_and_preload_rules_changed=False)
        result['sole_right_ownership_requires_no_giver_contact']=True
        result['contact_motion_retiming']={side:retimer.summary() for side,retimer in self.contact_retimers.items()}
        return result


class PhaseRuntime:
    """Explicit adapter adding giver-support evidence to the existing snapshot."""
    def __init__(self,base):self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    def snapshot(self,*args,**kwargs):
        snapshot=self.base.snapshot(*args,**kwargs)
        records=kwargs.get('records',args[-1] if args else None)
        frames=records.get('control_frame',[]);support=dict(snapshot.previous_control_frame_support or {})
        if frames:
            n=1
            while n<len(frames) and frames[-1-n]==frames[-1]:n+=1
            support['giver_contact_present']=any(np.any(np.asarray(records.get(name,[])[-n:])>=self.controller.primitive.force_threshold_n)
                for name in ('left_thumb_force_n','left_index_force_n','left_middle_force_n','left_palm_force_n'))
        return replace(snapshot,previous_control_frame_support=support or None)


def build_runtime(command_path,policy_safe_command,joint_names):
    runtime=original_runtime(command_path,policy_safe_command,joint_names)
    with np.load(command_path,allow_pickle=False) as a:
        if 'giver_release_not_before_frame' not in a:return runtime
        boundary=int(a['giver_release_not_before_frame'])
        policy=str(a['giver_release_policy']) if 'giver_release_policy' in a else 'simultaneous'
        coordinated=bool(a['coordinated_giver_release']) if 'coordinated_giver_release' in a else False
        release_frames=int(a['coordinated_release_frames']) if 'coordinated_release_frames' in a else None
        receiver_frames=int(a['receiver_transition_frames']) if 'receiver_transition_frames' in a else None
        delayed_digits=a['final_release_delayed_digits'].tolist() if 'final_release_delayed_digits' in a else []
        delayed_begin=int(a['final_release_delayed_begin']) if 'final_release_delayed_begin' in a else None
    old=runtime.controller;lower,upper,names=authoritative_joint_limits(read_json(JOINT_CONTRACT));assert list(names)==list(joint_names)
    runtime.controller=PhaseClockDex3(old.primitive,old.intent,old.raw,old.safe,'ACT-A40',lower[:14],upper[:14],lower[14:],upper[14:],release_not_before_frame=boundary,giver_release_policy=policy,coordinated_giver_release=coordinated,coordinated_release_frames=release_frames,receiver_transition_frames=receiver_frames,final_release_delayed_digits=delayed_digits,final_release_delayed_begin=delayed_begin)
    return PhaseRuntime(runtime)
