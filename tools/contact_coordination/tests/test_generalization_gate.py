import json
from pathlib import Path
import tempfile
import unittest
from tools.contact_coordination.generalization_gate import require, DOWNSTREAM


class GeneralizationGateTests(unittest.TestCase):
    def test_every_downstream_stage_fails_closed_without_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            for stage in DOWNSTREAM:
                with self.assertRaisesRegex(RuntimeError, 'GENERALIZATION_GATE_CLOSED'):
                    require(Path(folder), stage)

    def test_repairs_remain_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            for stage in ('prototype', 'TRAIN_pilot', 'target_repair', 'status'):
                require(Path(folder), stage)

    def test_ready_boolean_cannot_bypass_missing_physical_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'bootstrap').mkdir()
            (p/'SPLIT_CONTRACT.json').write_text('{}')
            (p/'bootstrap/SELECTION.json').write_text(json.dumps({'additional_train_source_ids':['one','two','three','four','five']}))
            value=dict(GENERALIZATION_REPAIR_READY='YES',golden_common_batch_pass=True,
                source_specific_targets_verified=True,no_known_common_registration_cache_contact_bug=True,
                non_golden_complete_valid_plan_source_ids=['one','two'],
                non_golden_physical_beyond_lift_source_ids=[],verified_evidence=[])
            (p/'GENERALIZATION_REPAIR_READY.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'GENERALIZATION_GATE_CLOSED'):
                require(p,'matched_dataset_realization')


if __name__=='__main__':unittest.main()
