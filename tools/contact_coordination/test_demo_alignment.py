"""Independent timing and missing-evidence regression examples."""
import unittest
import numpy as np
from .demo_alignment import check_arrays, scored_task_success
from .source_phase import acquisition_window


class TimingTests(unittest.TestCase):
    def test_effective_float32_supervision_detects_shift_and_command_as_state(self):
        from .target_dataset_audit import numeric_alignment
        observed=dict(executed_command=np.arange(112).reshape(4,28)/71.,
                      measured_state=np.arange(112).reshape(4,28)/137.,timestamp_s=np.arange(4)/30)
        raw={k:np.asarray(observed[v],dtype=np.float32) for k,v in
             [('action','executed_command'),('observation.state','measured_state'),('timestamp','timestamp_s')]}
        self.assertEqual(len(numeric_alignment(raw,observed)),2)
        with self.assertRaisesRegex(ValueError,'Effective tensor mismatch: action'):
            numeric_alignment(dict(raw,action=np.roll(raw['action'],1,axis=0)),observed)
        with self.assertRaisesRegex(ValueError,'Effective tensor mismatch: observation.state'):
            numeric_alignment({**raw,'observation.state':raw['action']},observed)

    def test_not_demonstrated_enum_cannot_become_true(self):
        score=dict(source_conditioned_full_task='NOT_DEMONSTRATED',stages={'FULL_TASK':False},physical_validity=True)
        self.assertIs(scored_task_success(score),False)
        score['stages']['FULL_TASK']=True
        with self.assertRaisesRegex(ValueError,'Inconsistent'):
            scored_task_success(score)

    def test_measured_pre_action_state_is_not_a_lagged_command(self):
        # Simulate a loaded servo that follows only part of its commanded step.
        actions = np.ones((3, 28)) * np.arange(1, 4)[:, None]
        measured = np.ones((24, 28)) * np.linspace(0., .6, 24)[:, None]
        obs = dict(control_frame=np.arange(3), timestamp_s=np.arange(3) / 30.,
                   measured_state=np.vstack([np.zeros(28), measured[7], measured[15]]),
                   executed_command=actions)
        trace = dict(control_frame=np.repeat(np.arange(3), 8), MEASURED_Q=measured,
                     EXECUTED_COMMAND=np.repeat(actions, 8, axis=0))
        self.assertEqual(check_arrays(obs, trace)['frames'], 3)
        wrong = {**obs, 'measured_state': np.vstack([np.zeros(28), actions[:2]])}
        with self.assertRaisesRegex(ValueError, 'Pre-action state'):
            check_arrays(wrong, trace)
        wrong = {**obs, 'executed_command': np.roll(actions, 1, axis=0)}
        with self.assertRaisesRegex(ValueError, 'actual servo command'):
            check_arrays(wrong, trace)

    def test_collapsed_close_event_requires_observed_closing(self):
        times = np.arange(6) / 30.
        events = dict(APPROACH_START=0., LEFT_CLOSE_BEGIN=times[5], LEFT_LIFT_BEGIN=times[5])
        state = np.zeros((6, 14)); state[:, 6] = [.05, .05, .04, .03, .02, .01]
        rows, correction = acquisition_window(times, events, state)
        np.testing.assert_array_equal(rows, [1, 2, 3, 4])
        self.assertEqual(correction['lift_begin_unchanged_s'], times[5])
        state[:, 6] = .05
        with self.assertRaisesRegex(ValueError, 'SOURCE_EVIDENCE_MISSING'):
            acquisition_window(times, events, state)
        events['LEFT_CLOSE_BEGIN'] = times[3]
        rows, correction = acquisition_window(times, events, state)
        np.testing.assert_array_equal(rows, [3, 4])
        self.assertIsNone(correction)


if __name__ == '__main__':
    unittest.main()
