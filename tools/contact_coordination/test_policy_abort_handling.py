import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from .act_policy_runtime import PolicyRuntime
from .policy_study import zero_transition_outcome
from .io import read,atomic_json


class PolicyAbortTests(unittest.TestCase):
    def runtime(self,folder,raw,project):
        runtime=PolicyRuntime.__new__(PolicyRuntime)
        runtime.folder=folder;runtime.rgb=lambda:np.zeros((480,640,3),np.uint8)
        runtime.bridge=SimpleNamespace(infer=lambda rgb,q:(raw,np.zeros(28),None,None,dict(status='UNIT_TEST_STUB')))
        runtime.projector=SimpleNamespace(hard_lower=np.full(28,-1.),hard_upper=np.full(28,1.),project=project)
        runtime.executed=np.zeros(28);runtime.rows=[];runtime.aborted=None
        return runtime

    def test_explicit_unsafe_arm_action_is_not_executed(self):
        with tempfile.TemporaryDirectory() as d:
            raw=np.zeros(28);raw[0]=2.
            runtime=self.runtime(Path(d),raw,lambda *a,**k:self.fail('Unsafe arm action reached projector'))
            runtime.step(0,dict(q=np.zeros(28)))
            row=read(Path(d)/'policy_steps/000000.json')
            self.assertIsNone(row['executed_command']);self.assertEqual(row['raw_action'][0],2.)
            self.assertEqual(runtime.aborted['status'],'POLICY_ADAPTER_SAFETY_ABORT')

    def test_internal_exception_remains_infrastructure_with_raw_output(self):
        with tempfile.TemporaryDirectory() as d:
            def broken(*a,**k):raise TypeError('demonstrated shared projector bug')
            runtime=self.runtime(Path(d),np.zeros(28),broken)
            with self.assertRaisesRegex(TypeError,'shared projector bug'):runtime.step(0,dict(q=np.zeros(28)))
            row=read(Path(d)/'policy_steps/000000.json')
            self.assertEqual(row['infrastructure_error']['type'],'TypeError');self.assertIsNone(runtime.aborted)
            self.assertEqual(len(row['raw_action']),28);self.assertIsNone(row['executed_command'])

    def test_initial_safety_abort_has_no_fabricated_physical_stages(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d);abort=dict(frame=0,reason='POLICY_ARM_COMMAND_OUTSIDE_NAMED_HARD_LIMITS')
            atomic_json(f/'ZERO_TRANSITION_POLICY_ABORT.json',dict(executed_control_frames=0))
            atomic_json(f/'DIRECT_EXECUTION_RUNTIME_SUMMARY.json',dict(aborted=abort,executed_commands=0))
            row=zero_transition_outcome(f)
            self.assertFalse(row['full_task_success']);self.assertFalse(row['unknown']);self.assertFalse(row['valid_physical_rollout'])
            self.assertEqual(row['stages'],{});self.assertEqual(row['executed_control_frames'],0)


if __name__=='__main__':unittest.main()
