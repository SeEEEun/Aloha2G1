"""Source-event-clock adapter for the common measured-contact Dex3 controller.

No representation/method/episode identifier is consumed by this controller.
Geometry, force/debounce/preload/limit rules remain the existing common ones.
"""
from dataclasses import replace
from pathlib import Path
import os
import numpy as np
from tools.direct_physical_execution_layer import DirectPhysicalDex3ExecutionLayer,authoritative_joint_limits
from tools.direct_physical_execution_isaac_runtime import DirectIsaacExecutionRuntime,verify_direct_freeze,JOINT_CONTRACT
from tools.common_execution_isaac_runtime import PHYSICS_CONFIG,PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER,WHOLE_HAND_GEOMETRY
from tools.common_execution_layer import Dex3Primitive,read_json,sha256_file

class SourceClockDex3(DirectPhysicalDex3ExecutionLayer):
    def __init__(self,primitive,intent,raw,safe,lower,upper,events,nominal_hand):
        # Constant legacy metadata token is identical for every invocation and
        # cannot identify a representation to the inherited numerical code.
        super().__init__(primitive,intent,raw,safe,'ACT-A40',lower[:14],upper[:14],lower[14:],upper[14:])
        # NPZ int64 frame indices have identical integer semantics, but must
        # cross the JSON summary interface as native Python integers.
        self.source_events={str(k):int(v) for k,v in events.items()};self.nominal_hand=np.asarray(nominal_hand)
        assert self.nominal_hand.shape==(len(safe),14)

    def _side_primitive(self,side):
        e=self.source_events
        begin=e['APPROACH_START'] if side=='left' else e['RIGHT_APPROACH_BEGIN']
        close=e['LEFT_CLOSE_BEGIN'] if side=='left' else e['RIGHT_CLOSE_BEGIN']
        complete=e['LEFT_CLOSE_COMPLETE'] if side=='left' else e['RIGHT_ACQUIRE_SOURCE']
        return replace(self.primitive,preshape_frames=max(1,close-begin),close_frames=max(1,complete-close))

    def _advance_grasp_state(self,side,*args,**kwargs):
        original=self.primitive;self.primitive=self._side_primitive(side)
        try:return super()._advance_grasp_state(side,*args,**kwargs)
        finally:self.primitive=original

    def _contact_seek(self,side,*args,**kwargs):
        original=self.primitive;self.primitive=self._side_primitive(side)
        try:return super()._contact_seek(side,*args,**kwargs)
        finally:self.primitive=original

    def _left_target(self,frame):
        # Preserve mechanical receiver-before-giver release. Failure to acquire
        # is physical data, never arm rescue or a source-clock rewrite.
        if frame>=self.source_events['LEFT_RELEASE_BEGIN'] and self.left_release_frame is None:
            return self.left_full_close.copy()
        return self.nominal_hand[frame,:7].copy()

    def _right_target(self,frame):return self.nominal_hand[frame,7:].copy()

    def step(self,frame,snapshot):
        e=self.source_events
        if self.left_trigger is None and frame>=e['APPROACH_START']:
            self.left_trigger=frame;self.left_start_q=self.left_open.copy()
        if self.right_trigger is None and frame>=e['RIGHT_APPROACH_BEGIN']:
            self.right_trigger=frame;self.right_start_q=self.right_open.copy()
        original=self.primitive
        # Do not allow the old controller to release the giver before the
        # source release phase, even if mechanical support is confirmed early.
        if frame<e['LEFT_RELEASE_BEGIN']:
            self.primitive=replace(original,right_verification_frames=len(self.safe)+1)
        try:return super().step(frame,snapshot)
        finally:self.primitive=original

    def summary(self):
        result=super().summary();result.pop('method',None)
        result.update(source_event_frames=self.source_events,controller='COMMON_SOURCE_CLOCK_DEX3',
            source_event_clock_changed=False,method_identifier_consumed=False,
            extra_legacy_preshape_close_delay_removed=True,mechanical_release_delay_is_outcome=True)
        return result

def build_runtime(command_path,policy_safe_command,joint_names):
    freeze=Path(os.environ['DIRECT_EVAL35_FREEZE_MANIFEST']);verify_direct_freeze(freeze)
    with np.load(command_path,allow_pickle=False) as z:
        names=z['joint_names'].astype(str).tolist();safe=z['commanded_q_rad'];raw=z['raw_policy_command'];intent=z['common_task_intent'].astype(str)
        initial=z['common_initial_q_rad'];events=dict(zip(z['source_event_names'].astype(str),z['execution_event_frames'].astype(int)))
        nominal=z['common_source_nominal_dex3']
    assert names==list(joint_names) and np.array_equal(safe,policy_safe_command)
    primitive=Dex3Primitive.from_frozen_dependencies(read_json(PHYSICS_CONFIG),read_json(PHYSICAL_ENVIRONMENT),read_json(COMMON_PHYSICAL_CONTROLLER))
    lower,upper,names2=authoritative_joint_limits(read_json(JOINT_CONTRACT));assert list(names2)==names
    controller=SourceClockDex3(primitive,intent,raw,safe,lower,upper,events,nominal)
    assert np.array_equal(initial[:14],safe[0,:14])
    assert np.allclose(initial[14:],np.r_[controller.left_open,controller.right_open],atol=1e-7,rtol=0)
    return DirectIsaacExecutionRuntime(controller=controller,joint_names=names,whole_hand_geometry=read_json(WHOLE_HAND_GEOMETRY),
        evaluator_manifest_sha256='NO_CLASSIFIER_OR_ATLAS',execution_manifest_sha256=sha256_file(freeze),
        direct_freeze_sha256=sha256_file(freeze),initial_q_rad=initial)
