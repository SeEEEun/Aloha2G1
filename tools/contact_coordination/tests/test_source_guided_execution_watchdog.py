import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.contact_coordination.io import atomic_json,atomic_npz,read,record
from tools.contact_coordination.source_guided_execution_watchdog import configure,preserve_incomplete


class ExecutionWatchdogTests(unittest.TestCase):
    def test_common_wall_budget_preserves_every_simulator_argument_and_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);command=['simulator','--dt','0.0041666667']
            atomic_json(folder/'INVOCATION.json',dict(command=command,timeout_s=1200))
            atomic_npz(folder/'input/COMMANDS.npz',stage=np.repeat('TRANSPORT',2400),q=np.zeros((2400,28)))
            before=record(folder/'input/COMMANDS.npz');receipt=configure(folder)
            self.assertEqual(read(folder/'INVOCATION.json')['command'],command)
            self.assertEqual(read(receipt)['timeout_s'],4920)
            self.assertEqual(record(folder/'input/COMMANDS.npz'),before)

    def test_completed_failed_task_is_never_repeated_or_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);folder=out/'trial'
            atomic_json(folder/'INVOCATION.json',dict(command=['simulator'],timeout_s=1200))
            atomic_json(folder/'PROCESS.json',dict(returncode=0))
            atomic_json(folder/'ABC_NOMINAL_SCORE.json',dict(stages=dict(FULL_TASK=False)))
            before=record(folder/'INVOCATION.json')
            self.assertIsNone(configure(folder));self.assertIsNone(preserve_incomplete(folder,out))
            self.assertEqual(record(folder/'INVOCATION.json'),before)
            self.assertFalse(read(folder/'ABC_NOMINAL_SCORE.json')['stages']['FULL_TASK'])

    def test_timeout_evidence_archived_before_fresh_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);folder=out/'physical_attempts/trial';folder.mkdir(parents=True)
            (folder/'engine.log').write_text('HYBRID_MEASURED_PROGRESS 1800 2342\n')
            atomic_json(folder/'PROCESS.json',dict(returncode='TIMEOUT'))
            receipt=preserve_incomplete(folder,out)
            self.assertFalse(folder.exists());self.assertTrue(receipt.exists())
            self.assertEqual((receipt.parent/'engine.log').read_text(),'HYBRID_MEASURED_PROGRESS 1800 2342\n')
            self.assertFalse(read(receipt)['official_physical_task_result'])


if __name__=='__main__':unittest.main()
