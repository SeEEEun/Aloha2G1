import sys,unittest,json,csv
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
from diagnose_g1_collision_root_cause import classify
class TestRootCause(unittest.TestCase):
 def test_wrist_finger(self):self.assertEqual(classify('right_wrist_yaw_link','right_hand_thumb_2_link'),'same-hand wrist–finger')
 def test_hand_torso(self):self.assertEqual(classify('torso_link','left_hand_index_1_link'),'hand–torso')
 def test_hand_hand(self):self.assertEqual(classify('left_hand_middle_1_link','right_hand_middle_1_link'),'hand–hand')
 def test_cross(self):self.assertEqual(classify('left_elbow_link','right_hand_index_1_link'),'cross-arm–hand')
 def test_preservation_output(self):
  p=ROOT/'outputs/collision_root_cause/diagnostic_variants.npz'
  if not p.exists():self.skipTest('run diagnostic first')
  with np.load(p) as z:
   ids=[list(z['full_joint_names']).index(n) for n in z['arm_joint_names']]
   self.assertTrue(np.array_equal(z['current_full_qpos'][:,ids],z['arm_qpos']))
   self.assertTrue(np.array_equal(z['all_fingers_open_qpos'][:,ids],z['arm_qpos']))
   self.assertTrue(np.array_equal(z['position_only_qpos'],z['current_full_qpos']))
   with np.load(ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz') as src:self.assertTrue(np.array_equal(z['current_full_qpos'],src['full_qpos']))
 def test_all_open_fingers(self):
  p=ROOT/'outputs/collision_root_cause/diagnostic_variants.npz';cfg=json.loads((ROOT/'configs/dex3_magsafe_grasp_primitives.sim.json').read_text())
  with np.load(p) as z:
   for side,key in (('left','LEFT_PHONE_OPEN'),('right','RIGHT_ACCESSORY_OPEN')):
    names=cfg['joint_names'][side+'_dex3'];ids=[list(z['full_joint_names']).index(n) for n in names];want=np.asarray(cfg['primitives'][key]['qpos']);self.assertTrue(np.array_equal(z['all_fingers_open_qpos'][:,ids],np.repeat(want[None],len(z['timestamps']),axis=0)))
 def test_current_reproduced(self):
  s=json.loads((ROOT/'outputs/collision_root_cause/root_cause_summary.json').read_text());self.assertEqual(s['current_vs_all_open']['current']['records'],279);self.assertEqual(s['current_vs_all_open']['current']['unique_frames'],128)
 def test_contact_visualization(self):
  for f in (224,251,285,298):
   for view in ('front','side','top'):self.assertTrue((ROOT/f'outputs/collision_root_cause/frames/{f}/collision_{view}.png').exists())
 def test_no_robot_api(self):
  s=(ROOT/'tools/diagnose_g1_collision_root_cause.py').read_text()
  for forbidden in ('ChannelPublisher','ChannelFactoryInitialize','xr_teleop','rt/lowstate'):self.assertNotIn(forbidden,s)
if __name__=='__main__':unittest.main()
