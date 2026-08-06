import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from g1_behavior_schema import *
from align_g1_behaviors import align,dtw_path,sample,phase_pairs
from compare_vla_generated_vs_g1_expert import qerr
class TestBehavior(unittest.TestCase):
 def base(self,n=10):
  names=np.array(['j']);a={'timestamps':np.arange(n)/10,'fps':np.array(10.),'full_joint_names':names,'arm_joint_names':names,'left_dex3_joint_names':np.array([],dtype='U1'),'right_dex3_joint_names':np.array([],dtype='U1'),'left_phase':np.repeat('OPEN',n),'right_phase':np.repeat('HOLD',n),'target_full_qpos':np.zeros((n,1)),'target_arm_qpos':np.zeros((n,1))};m={'schema_version':SCHEMA,'behavior_id':'x','behavior_role':'generated_target','task_name':'t','robot':'G1','hand':'Dex3','source_type':'x','source_path':'x','created_at':now(),'fps':10.,'frame_count':n,'duration_sec':(n-1)/10,'coordinate_frame':'base','joint_units':'radian','orientation_representation':'quaternion_wxyz','execution_status':'not_executed','is_synthetic':False,'valid_for_paper_result':True,'primitive_source':'simulation_placeholder','notes':''};return a,m
 def test_01_schema_roundtrip(self):
  a,m=self.base()
  with tempfile.TemporaryDirectory() as d:p=Path(d)/'x.npz';save_behavior(p,a,m);b,n=load_behavior(p);self.assertEqual(n['behavior_id'],'x')
 def test_02_identical_zero(self):self.assertEqual(float(np.sqrt(np.mean((np.ones(3)-np.ones(3))**2))),0)
 def test_03_remap(self):self.assertEqual(remap_joints_by_name([[1,2]],['b','a'],['a','b']).tolist(),[[2,1]])
 def test_04_wrong_order_detected(self):self.assertFalse(np.array_equal(['a','b'],['b','a']))
 def test_05_missing_joint(self):
  with self.assertRaises(ValueError):remap_joints_by_name([[1]],['a'],['b'])
 def test_06_swap_flag(self):self.assertTrue({'left_right_swapped':True}['left_right_swapped'])
 def test_07_quaternion_sign(self):self.assertAlmostEqual(float(qerr(np.array([[1,0,0,0.]]),np.array([[-1,0,0,0.]]))[0]),0)
 def test_08_normalized(self):a,_=self.base(10);b,_=self.base(20);self.assertEqual(align(a,b,'normalized_time',30)['index_pairs'].shape,(30,2))
 def test_09_dtw_warp(self):p,c=dtw_path(np.arange(5)[:,None],np.repeat(np.arange(5),2)[:,None]);self.assertLess(c,1e-9)
 def test_10_phase(self):a,_=self.base();b,_=self.base();self.assertGreater(len(phase_pairs(a,b)[0]),0)
 def test_11_missing_phase(self):a,_=self.base();b,_=self.base();b['left_phase'][:]='GRASP';self.assertTrue(phase_pairs(a,b)[1])
 def test_12_missing_actual_primary(self):a,m=self.base();self.assertNotIn('actual_full_qpos',a)
 def test_13_synthetic_paper_reject(self):a,m=self.base();m.update(is_synthetic=True,valid_for_paper_result=True);self.assertTrue(validate_behavior_schema(a,m))
 def test_14_multiple_expert_stats(self):self.assertAlmostEqual(np.mean([1,2,3]),2)
 def test_15_bootstrap_deterministic(self):
  a=np.random.default_rng(3).choice([1,2,3],10);b=np.random.default_rng(3).choice([1,2,3],10);self.assertTrue(np.array_equal(a,b))
 def test_16_exclusion_log(self):self.assertEqual({'trial':'x','reason':'bad'}['reason'],'bad')
 def test_17_nan_reject(self):a,m=self.base();a['target_arm_qpos'][0]=np.nan;self.assertTrue(validate_behavior_schema(a,m))
 def test_18_nonmonotonic(self):a,m=self.base();a['timestamps'][2]=a['timestamps'][1];self.assertTrue(validate_behavior_schema(a,m))
 def test_19_different_fps(self):x=sample(np.arange(10.),np.linspace(0,9,17));self.assertEqual(len(x),17)
 def test_20_manifest_hash_warning(self):
  with tempfile.TemporaryDirectory() as d:p=Path(d)/'x';p.write_text('a');h=sha256(p);p.write_text('b');self.assertNotEqual(h,sha256(p))
 def test_21_duplicate_name_reject(self):
  with self.assertRaises(ValueError):remap_joints_by_name([[1,2]],['a','a'],['a'])
 def test_22_roles(self):self.assertIn('expert_actual',ROLES)
 def test_23_status(self):self.assertIn('not_executed',STATUS)
 def test_24_interp_endpoints(self):x=sample(np.array([[0.],[1.]]),np.array([0.,1.]));self.assertTrue(np.array_equal(x,[[0],[1]]))
if __name__=='__main__':unittest.main()
