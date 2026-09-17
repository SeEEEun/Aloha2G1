import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .abc_contract import METHODS, require, eligible_supervision
from .io import atomic_json,read
from .full_attempt_runtime import FullAttemptRuntime


class Delegate:
    initial_q_rad=np.zeros(28)
    event_field_names=()
    def __init__(self):
        self.calls=[];self.controller=SimpleNamespace(grasp_confirmed_frame={'left':None},grasp_state={'left':'OPEN'})
    def step(self,frame,snapshot):
        self.calls.append(frame);return np.full(28,frame/100.)
    def event_values(self,measured):return {'MEASURED_Q':measured}
    def write_summary(self,path):atomic_json(path,{})


class ABCContractTests(unittest.TestCase):
    def test_only_nonfinite_measurements_abort_recording(self):
        from .full_attempt_runtime import state_validity_reason
        gates=dict(maximum_object_linear_speed_m_s=1.,maximum_object_angular_speed_rad_s=50.,maximum_object_com_step_m=.03)
        pose=np.r_[np.zeros(3),0,0,0,1];v=np.zeros(6)
        self.assertIsNone(state_validity_reason(pose,v,np.zeros(28),np.zeros(28),np.zeros(3),gates))
        v[2]=-1.5
        self.assertIsNone(state_validity_reason(pose,v,np.zeros(28),np.zeros(28),np.zeros(3),gates))
        moved=pose.copy();moved[0]=.031
        self.assertIsNone(state_validity_reason(moved,v,np.zeros(28),np.zeros(28),np.zeros(3),gates))
        v[3]=51.;self.assertIsNone(state_validity_reason(pose,v,np.zeros(28),np.zeros(28),np.zeros(3),gates))
        v[3]=float('nan');self.assertEqual(state_validity_reason(pose,v,np.zeros(28),np.zeros(28),np.zeros(3),gates),'NONFINITE_MEASURED_STATE')
        for field in range(4):
            values=[pose.copy(),np.zeros(6),np.zeros(28),np.zeros(28)]
            values[field][0]=float('inf')
            self.assertEqual(state_validity_reason(*values,np.zeros(3),gates),'NONFINITE_MEASURED_STATE')

    def test_falling_object_speed_warning_preserves_full_recording_and_latch(self):
        from .full_attempt_runtime import state_validity_reason,record_linear_speed_telemetry
        gates=dict(maximum_object_linear_speed_m_s=1.,maximum_object_angular_speed_rad_s=50.,maximum_object_com_step_m=.03)
        velocity=np.zeros((10,3));velocity[:,2]=[0.,-.2,-.4,-.6,-.8,-1.,-1.2,-1.4,-1.6,-1.8]
        timestamp=np.arange(10,dtype=np.float64)/240.
        for channel in ['OFFICIAL_NOMINAL','DIAGNOSTIC_FULL_CONTINUATION','RECONSTRUCTED_DIAGNOSTIC']:
            with self.subTest(channel=channel), tempfile.TemporaryDirectory() as folder:
                runtime=self.runtime(folder,2,channel)
                for frame in range(10):
                    self.assertIsNone(state_validity_reason(np.r_[np.zeros(3),0,0,0,1],np.r_[velocity[frame],np.zeros(3)],np.zeros(28),np.zeros(28),np.zeros(3),gates))
                    runtime.step(frame,None)
                angular=np.zeros((10,3));angular[7,0]=55.
                position=np.zeros((10,3));position[8:,0]=.04
                trace=dict(object_linear_velocity_m_s=velocity.copy(),object_angular_velocity_rad_s=angular.copy(),
                    object_position_world_m=position.copy(),timestamp_s=timestamp.copy(),physics_step=np.arange(10),control_frame=np.arange(10))
                telemetry=record_linear_speed_telemetry(trace,gates,folder)
                np.testing.assert_array_equal(trace['object_linear_velocity_m_s'],velocity)
                np.testing.assert_array_equal(trace['timestamp_s'],timestamp)
                np.testing.assert_array_equal(trace['object_linear_speed_warning'],[False]*6+[True]*4)
                self.assertEqual(telemetry['maximum']['speed_m_s'],1.8)
                self.assertEqual(telemetry['maximum']['timestamp_s'],timestamp[-1])
                self.assertEqual(telemetry['first_exceedance']['timestamp_s'],timestamp[6])
                self.assertFalse(telemetry['terminal_condition'])
                validity=read(Path(folder)/'FINITE_STATE_VALIDITY_TELEMETRY.json')
                self.assertFalse(validity['terminal_condition'])
                self.assertTrue(validity['scoring_thresholds_unchanged'])
                self.assertEqual(validity['checks']['angular_speed']['first_exceedance']['control_frame'],7)
                self.assertEqual(validity['checks']['com_step']['first_exceedance']['control_frame'],8)
                np.testing.assert_array_equal(trace['object_angular_velocity_rad_s'],angular)
                np.testing.assert_array_equal(trace['object_position_world_m'],position)
                self.assertEqual(runtime.frame,9)
                self.assertEqual(runtime.delegate.calls,list(range(8)))
                self.assertEqual(runtime.first_failure,dict(reason='INJECTED_NONFATAL_RECORDING_TEST',control_frame=2))
                self.assertIsNone(runtime.stop)

    def test_parameter_defaults_preserve_backend_and_reject_expansion(self):
        from .calibration_parameters import parameters,DEFAULT,scaled_ik_config
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);cfg=dict(joint_prior_scale=.001,position_residual_scale=100.)
            self.assertEqual(scaled_ik_config(p,cfg,'acquisition'),cfg)
            atomic_json(p/'CALIBRATION_PARAMETERS.json',dict(values=DEFAULT,plotting={'dpi':300}))
            self.assertEqual(parameters(p),DEFAULT)
            changed=dict(DEFAULT,receiver_spread_fraction=1.01)
            atomic_json(p/'CALIBRATION_PARAMETERS.json',dict(values=changed))
            with self.assertRaises(ValueError):parameters(p)
    def test_aliases_do_not_relabel_historical_coupled_as_independent(self):
        self.assertEqual(METHODS['C_COUPLED']['legacy_key'],'B')
        self.assertEqual(METHODS['B_INDEPENDENT']['legacy_key'],'B_NO_COUPLING')

    def test_all_calibration_and_final_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);atomic_json(p/'ABC_STUDY.json',{})
            for stage in ['TRAIN_calibration','bounded_calibration','TRAIN40_ABC','ACT_BC','ACT_DEV35',
                          'TRAIN40_conversion','ACT_train_or_verified_reuse','reference_coupling10']:
                with self.assertRaisesRegex(RuntimeError,'ABC_GATE_CLOSED'):require(p,stage)
            require(p,'TRAIN_control')

    def runtime(self,folder,injected=2,channel='OFFICIAL_NOMINAL',stages=None):
        return FullAttemptRuntime(Delegate(),dict(output_dir=folder,nominal_frames=8,observation_s=2.,
            evidence_channel=channel,contract_test_only=True,injected_nonfatal_failure_frame=injected,
            source_id='SYNTHETIC',method_key='C_COUPLED'),stages or ['OPEN']*10)

    def test_nonfatal_flag_timestamp_cannot_shorten_or_change_commands(self):
        with tempfile.TemporaryDirectory() as f:
            a=self.runtime(f,2);b=self.runtime(f,6)
            aa=[a.step(i,None) for i in range(10)];bb=[b.step(i,None) for i in range(10)]
            np.testing.assert_array_equal(aa,bb)
            self.assertEqual(a.delegate.calls,list(range(8)));self.assertEqual(a.frame,b.frame)
            self.assertNotEqual(a.first_failure,b.first_failure)
            a.write_summary(Path(f)/'delegate.json')
            from .io import read
            self.assertFalse(read(Path(f)/'FULL_ATTEMPT_RECORDING.json')['eligible_for_official_scoring'])

    def test_official_failure_preserves_remaining_commands_and_recording_horizon(self):
        with tempfile.TemporaryDirectory() as f:
            r=self.runtime(f,-1,stages=['OPEN']*2+['LIFT_5CM']*8)
            q=[r.step(i,None) for i in range(10)]
            self.assertEqual(r.delegate.calls,list(range(8)));self.assertEqual(r.frame,9)
            self.assertEqual(r.first_failure['control_frame'],2);self.assertIsNone(r.stop)
            self.assertGreater(np.max(q[7]-q[1]),0.)
            self.assertFalse(r.event_values(np.zeros(28))['CONTROL_ADMISSION_STOPPED'])

    def test_diagnostic_advances_without_becoming_nominal_evidence(self):
        with tempfile.TemporaryDirectory() as f:
            r=self.runtime(f,-1,'DIAGNOSTIC_FULL_CONTINUATION',['OPEN']*2+['LIFT_5CM']*8)
            q=[r.step(i,None) for i in range(10)]
            self.assertEqual(r.delegate.calls,list(range(8)))
            self.assertGreater(np.max(q[7]-q[1]),0)
            self.assertTrue(r.interventions);self.assertIsNone(r.stop)
            row=dict(evidence_channel=r.channel,physical_validity=True,complete_command_sequence=True,
                     observation_action_alignment_verified=True,task_success=True)
            self.assertFalse(eligible_supervision(row))
            row['evidence_channel']='OFFICIAL_NOMINAL';row['task_success']=False
            self.assertTrue(eligible_supervision(row))


if __name__=='__main__':unittest.main()
