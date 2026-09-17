import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.contact_coordination.io import atomic_json,atomic_npz,record
from tools.contact_coordination.source_guided_physics_cache import same_commands,find_equivalent


class EquivalentExecutionTests(unittest.TestCase):
    def test_outer_stage_receipt_notices_changed_source_array(self):
        from tools.contact_coordination.run_source_guided_rrt_rebuild import dependencies,signature
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            atomic_json(out/'COVERAGE8.json',dict(golden='fixture',source_ids=[]))
            source=out/'source_phase/fixture/FUNCTIONAL_WRIST_PRIORS.npz'
            atomic_npz(source,wrist=np.zeros((2,3)))
            before=signature(dependencies(out,'golden_regression'))
            atomic_npz(source,wrist=np.ones((2,3)))
            self.assertNotEqual(before,signature(dependencies(out,'golden_regression')))

    def test_reuse_requires_exact_commands_scene_identity_and_dependency_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);old=out/'old_plan';new=out/'new_plan';folder=out/'physical_attempts/trial'
            q=np.zeros((2,28));stage=np.array(['START','END']);payload=dict(commanded_q_rad=q,stage=stage,giver_release_not_before_frame=np.array(1))
            for plan in (old,new):
                atomic_npz(plan/'COMMANDS.npz',**payload);atomic_json(plan/'PLAN.json',dict(source_id='fixture'))
                atomic_json(plan/'SOURCE_SCENE.json',dict(target_object_pose=dict(position=[.3,.1,.8])))
            atomic_json(folder/'input/SOURCE_SCENE.json',dict(target_object_pose=dict(position=[.3,.1,.8])))
            atomic_json(folder/'input/FULL_ATTEMPT_CONFIG.json',dict(source_id='fixture',method_key='B_INDEPENDENT',evidence_channel='OFFICIAL_NOMINAL',
                contract_test_only=False,source_commands=record(old/'COMMANDS.npz'),nominal_frames=2,observation_s=1.))
            atomic_json(out/'ABC_STUDY.json',dict(nominal_settle_observation_s=1.))
            code=out/'runtime.py';code.write_text('unchanged controller')
            atomic_json(folder/'DEPENDENCIES.json',dict(files=[record(code)]));atomic_json(folder/'PROCESS.json',dict(returncode=0))
            atomic_json(folder/'FULL_ATTEMPT_RECORDING.json',dict(recording_end_reason='COMPLETE_DECLARED_HORIZON',nominal_frames=2,requested_recording_frames=32))
            atomic_npz(folder/'event_log.npz',measured_q=np.zeros((32,28)))
            # A failed task is just as reusable as a successful one: the cache
            # does not inspect an outcome or choose a successful repetition.
            atomic_json(folder/'ABC_NOMINAL_SCORE.json',dict(stages=dict(FULL_TASK=False)))
            self.assertIsNotNone(find_equivalent(out,new,'B_INDEPENDENT'))
            self.assertIsNone(find_equivalent(out,new,'C_COUPLED'))
            atomic_npz(new/'COMMANDS.npz',**dict(payload,giver_release_not_before_frame=np.array(0)))
            self.assertFalse(same_commands(old/'COMMANDS.npz',new/'COMMANDS.npz'))
            self.assertIsNone(find_equivalent(out,new,'B_INDEPENDENT'))
            atomic_npz(new/'COMMANDS.npz',**payload)
            atomic_json(new/'SOURCE_SCENE.json',dict(target_object_pose=dict(position=[.31,.1,.8])))
            self.assertIsNone(find_equivalent(out,new,'B_INDEPENDENT'))
            atomic_json(new/'SOURCE_SCENE.json',dict(target_object_pose=dict(position=[.3,.1,.8])))
            code.write_text('changed controller')
            self.assertIsNone(find_equivalent(out,new,'B_INDEPENDENT'))


if __name__=='__main__':unittest.main()
