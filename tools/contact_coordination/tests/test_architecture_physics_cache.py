"""A completed process is not reusable evidence after a dependency changes."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from tools.contact_coordination.io import atomic_json,record
from tools.contact_coordination.full_attempt import launch


class PhysicsResumeTests(unittest.TestCase):
    def test_completed_receipt_still_checks_dependencies(self):
        with TemporaryDirectory() as directory:
            folder=Path(directory);dependency=folder/'controller.py';dependency.write_text('original')
            atomic_json(folder/'DEPENDENCIES.json',dict(files=[record(dependency)]))
            atomic_json(folder/'PROCESS.json',dict(returncode=0))
            self.assertEqual(launch(folder,resume=True)['returncode'],0)
            dependency.write_text('changed')
            with self.assertRaises(AssertionError):launch(folder,resume=True)


if __name__=='__main__':unittest.main()
