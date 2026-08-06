from __future__ import annotations
import importlib.util,signal,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'tools/record_aloha_readonly_state_lowlevel.py'
spec=importlib.util.spec_from_file_location('readonly_audit',SCRIPT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Fake:
 def __init__(self):self.calls=[]
 def __getattr__(self,name):
  def f(*a,**k):
   self.calls.append(name)
   if name in m.DENYLIST:raise AssertionError('mutating call '+name)
   return []
  return f
class TestReadonlyAudit(unittest.TestCase):
 def test_import_creates_no_object(self):self.assertFalse(m.audit_plan()['hardware_object_created'])
 def test_getter_only_fake(self):
  f=Fake();m.read_preconfigured_fake_for_test(f);self.assertEqual(set(f.calls),set(m.ALLOWLIST));self.assertFalse(set(f.calls)&set(m.DENYLIST))
 def test_default_dry_and_exception_have_no_mutation(self):
  for args in [[],['--audit-only'],['--dry-run'],['--print-plan']]:
   r=subprocess.run([sys.executable,str(SCRIPT),*args],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('"network_connection_opened": false',r.stdout)
 def test_live_gate_always_blocks(self):
  r=subprocess.run([sys.executable,str(SCRIPT),'--enable-live-readonly','--confirmed-no-command-path','--confirmed-driver-version','--confirmed-safe-cleanup','--operator-present'],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('BLOCKED_UNSAFE',r.stderr)
 def test_sigint_path_has_no_cleanup_code(self):
  source=SCRIPT.read_text();self.assertNotIn('signal.signal(',source);self.assertNotIn('.cleanup(',source);self.assertNotIn('.configure(',source)
if __name__=='__main__':unittest.main()
