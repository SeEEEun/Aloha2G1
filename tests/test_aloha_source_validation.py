from __future__ import annotations
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import aloha_source_validation_common as c
class TestValidation(unittest.TestCase):
 def test_bad_shape_and_nan(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.npz';np.savez(p,optimized_action=np.zeros((2,14)),fps=30)
   with self.assertRaises(ValueError):c.load_action(p)
   np.savez(p,optimized_action=np.full((990,14),np.nan),fps=30)
   with self.assertRaises(ValueError):c.load_action(p)
 def test_mapping_blocks_dimension_swap_duplicate(self):
  self.assertEqual(c.validate_mapping(c.NAMES[:-1])['status'],'BLOCK')
  x=c.NAMES.copy();x[:7],x[7:]=x[7:],x[:7];self.assertEqual(c.validate_mapping(x)['status'],'BLOCK')
  x=c.NAMES.copy();x[1]=x[0];self.assertEqual(c.validate_mapping(x)['status'],'BLOCK')
 def test_stale_and_frame_jump_primitives(self):
  cfg=json.loads((ROOT/'configs/aloha_source_validation_safety.unreviewed.json').read_text());self.assertLess(cfg['state_stale_seconds'],1)
  q=np.zeros(14);q2=q.copy();q2[0]=1;self.assertTrue(np.any(abs(q2-q)>cfg['max_step_per_channel']))
 def test_metrics_and_gripper_delay(self):
  x=np.zeros((100,14));x[20:,6]=1;y=np.zeros_like(x);y[3:]=x[:-3];m=c.tracking_metrics(x,y);self.assertAlmostEqual(m['left_gripper_event_delay_s'],.1,places=5);self.assertGreater(m['rmse'],0)
 def test_gate_and_dry_run_no_publisher(self):
  s=ROOT/'tools/replay_aloha_source_action_safely.py';r=subprocess.run([sys.executable,str(s),'dry-run','--end-frame','10','--trial-id','unit_test_dry'],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('"publisher_created": false',r.stdout)
  r=subprocess.run([sys.executable,str(s),'execute','--mode','segment'],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('missing gates',r.stderr)
 def test_fixture_provenance_distinct(self):
  self.assertNotEqual('SIM_FIXTURE_SYNTHETIC_NOT_REAL','REAL')
if __name__=='__main__':unittest.main()
