import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from extract_magsafe_gripper_phases import GripperPhaseTracker
from record_g1_dex3_magsafe_primitives import assert_real_robot_primitive_config

def track(x,**kw):
 t=GripperPhaseTracker(.7,.3,min_dwell_frames=2,grasp_transition_frames=2,**kw);return [t.update(v)['phase'] for v in x]
class TestPhases(unittest.TestCase):
 def test_constant_open(self):self.assertEqual(set(track([1]*10)),{'OPEN'})
 def test_open_close(self):self.assertIn('HOLD',track([1]*4+[.6,.4]+[.2]*8))
 def test_all_close_phases(self):self.assertTrue({'OPEN','PREGRASP','GRASP','HOLD'}<=set(track([1]*4+[.6,.4]+[.2]*8)))
 def test_release(self):self.assertTrue({'HOLD','RELEASE','OPEN'}<=set(track([0]*4+[.4,.6]+[.8]*5)))
 def test_noisy_crossing(self):self.assertNotIn('GRASP',track([1,1,.31,.29,.31,.29,.31]))
 def test_one_frame_spike(self):self.assertEqual(track([1,1,0,1,1])[-1],'OPEN')
 def test_independent(self):self.assertNotEqual(track([1]*8),track([0]*8))
 def test_identical_semantic_qpos_allowed(self):self.assertTrue(np.array_equal(np.zeros(7),np.zeros(7)))
 def test_missing_primitive(self):self.assertNotIn('X',{'OPEN':{}})
 def test_wrong_joint_order(self):self.assertNotEqual(['a','b'],['b','a'])
 def test_nan(self):
  with self.assertRaises(ValueError):track([1,np.nan])
 def test_mismatched_frames(self):self.assertNotEqual(len(np.zeros((2,14))),len(np.zeros((3,7))))
 def test_sim_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'SIMULATION PRIMITIVES'):assert_real_robot_primitive_config({'source':'simulation_placeholder','authoritative_for_real_robot':False,'real_robot_command_allowed':False})
 def test_arm_preservation(self):
  arm=np.arange(28).reshape(2,14);full=np.zeros((2,50));full[:,7:21]=arm;self.assertTrue(np.array_equal(full[:,7:21],arm))
 def test_state_roundtrip(self):
  t=GripperPhaseTracker(.7,.3);t.update(1);s=t.serialize_state();u=GripperPhaseTracker(.7,.3);u.load_state(s);self.assertEqual(t.get_state(),u.get_state())
if __name__=='__main__':unittest.main()
