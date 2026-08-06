import hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
R=Path('/home/jbnu/aloha_g1_dataset');O=R/'outputs/retargeting_method_comparison';M=('relative_temporal_proposed','relative_framewise','relative_no_workspace_scale')
class T(unittest.TestCase):
 def test_freeze(self):self.assertTrue(json.load(open(O/'proposed_freeze.json'))['exact_array_reproduction'])
 def test_source_hash(self):self.assertEqual(json.load(open(O/'proposed_freeze.json'))['source_action_sha256'],'a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c')
 def test_proposed_hash(self):self.assertEqual(json.load(open(O/'proposed_freeze.json'))['proposed_arm_sha256'],'c58c8ee6f98e02d71e22abc721fcb92bb7e5c233963b0cb2d44b3fa6c4ad1f3e')
 def test_fairness(self):self.assertEqual(json.load(open(O/'fairness_validation.json'))['status'],'PASS')
 def test_shapes(self):
  for m in M:
   with np.load(O/'trajectories'/m/'arm_trajectory.npz',allow_pickle=True) as z:self.assertEqual(z['arm_qpos'].shape,(990,14))
 def test_placeholder_equal(self):
  vals=[]
  for m in M:
   with np.load(O/'trajectories'/m/'full_trajectory_placeholder_dex3.npz') as z:vals.append((z['left_dex3_qpos'].copy(),z['right_dex3_qpos'].copy()))
  for x in vals[1:]:self.assertTrue(np.array_equal(vals[0][0],x[0]) and np.array_equal(vals[0][1],x[1]))
 def test_arm_embedded_exact(self):
  for m in M:
   with np.load(O/'trajectories'/m/'full_trajectory_placeholder_dex3.npz') as z:self.assertTrue(np.array_equal(z['arm_qpos'],z['full_qpos'][:,[list(z['full_joint_names']).index(n) for n in z['arm_joint_names']]]))
 def test_arm_level_separation(self):
  for m in M:self.assertTrue((O/'mujoco'/m/'arm_level_collision.json').exists() and (O/'mujoco'/m/'full_hand_diagnostic.json').exists())
 def test_isaac_finite(self):
  for m in M:self.assertEqual(json.load(open(O/'isaaclab'/m/'replay_summary.json'))['simulation_stability'],'FINITE')
 def test_absolute_not_fabricated(self):self.assertEqual(json.load(open(O/'trajectories/absolute_temporal/method_config.json'))['status'],'NOT_AVAILABLE')
 def test_aloha_command_disabled(self):self.assertFalse(json.load(open(R/'outputs/aloha_real_validation/dry_run_inspection.json'))['command_enabled'])
 def test_no_g1_api(self):
  for p in ('generate_retargeting_method_comparison.py','evaluate_retargeting_method_comparison.py','run_retargeting_method_comparison_offline.py'):
   s=(R/'tools'/p).read_text();self.assertNotIn('ChannelPublisher',s);self.assertNotIn('ChannelFactoryInitialize',s)
 def test_report_outputs(self):
  for p in ('paper_table_retargeting.csv','paper_table_isaac_tracking.csv','paper_table_placeholder_hand_diagnostic.csv','index.html','retargeting_methods_grid.mp4'):self.assertTrue((O/'report'/p).exists())
if __name__=='__main__':unittest.main()
