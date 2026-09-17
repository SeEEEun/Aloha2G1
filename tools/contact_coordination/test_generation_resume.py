"""Completed generation must resume without new searches or evidence writes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import generation_study


class CompletedResumeTest(unittest.TestCase):
    def test_all_eighty_saved_attempts_are_read_without_rerunning(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            area=out/'TRAIN40_conversion';area.mkdir()
            schedule=[dict(index=i,source_id=f'source_{i//2}',condition='AB'[i%2]) for i in range(80)]
            ledger=dict(status='FROZEN_TRAIN40_COMPLETE',scheduled=80,completed=80,rows=schedule)
            path=area/'LEDGER.json';path.write_text(json.dumps(ledger))
            original=path.read_bytes()
            with patch.object(generation_study,'verify_freeze',return_value=dict(schedule=schedule)), \
                 patch('tools.contact_coordination.episode_context.prepare',side_effect=AssertionError('must not prepare another attempt')), \
                 patch('tools.contact_coordination.conversion_attempt.attempt',side_effect=AssertionError('must not execute another attempt')), \
                 patch.object(generation_study.subprocess,'Popen',side_effect=AssertionError('must not launch a process')):
                self.assertEqual(generation_study.run(out,resume=True),ledger)
            self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':
    unittest.main()
