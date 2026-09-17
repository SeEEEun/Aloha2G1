import tempfile,unittest
from pathlib import Path
from tools.contact_coordination.io import read,record,atomic_json,atomic_text
from tools.contact_coordination.practical_planning import rank
from tools.contact_coordination.practical_results import episode


class PracticalRunnerTests(unittest.TestCase):
    def fixture(self,out,counts):
        names=['CURRENT_DEFAULT','LOWER_CLEARANCE','COMPACT_APPROACH','NEAR_APPROACH'];rows=[]
        atomic_json(out/'PRACTICAL_STUDY.json',dict(recipes=names))
        for name,pair in zip(names,counts,strict=True):
            for method,count in zip(['B_INDEPENDENT','C_COUPLED'],pair,strict=True):
                for i in range(40):
                    ok=i<count
                    rows.append(dict(recipe=name,method=method,source_id=f'source_{i:02}',planning_seconds=1.,
                        summary=dict(COMPLETE_PLAN=ok,NO_IK=not ok,NO_COMPLETE_CHAIN=0,source_deviation_m2=.001 if ok else None,
                            first_causal_failure=None if ok else dict(cause='NO_IK',phase='APPROACH_CLEARANCE'))))
        atomic_json(out/'PLANNING_SWEEP.json',dict(rows=rows))

    def test_all_four_flat_recipes_stop_before_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);self.fixture(out,[(8,8)]*4);rank(out);result=read(out/'RECIPE_RANKING.json')
            self.assertFalse(result['planning_improved']);self.assertEqual(result['top_recipes'],[])
            self.assertTrue((out/'PLANNING_CALIBRATION_INSUFFICIENT.md').exists())

    def test_equal_coverage_prefers_stronger_weaker_method_not_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);self.fixture(out,[(8,8),(10,10),(9,11),(8,12)]);rank(out);result=read(out/'RECIPE_RANKING.json')
            self.assertTrue(result['planning_improved']);self.assertEqual(result['ranked'][0]['recipe'],'LOWER_CLEARANCE')

    def test_one_new_source_is_not_meaningful(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);self.fixture(out,[(8,8),(9,9),(8,8),(8,8)]);rank(out)
            self.assertFalse(read(out/'RECIPE_RANKING.json')['planning_improved'])

    def test_mean_coverage_objective_has_no_extra_per_method_no_loss_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);self.fixture(out,[(10,10),(9,15),(10,10),(10,10)]);rank(out)
            result=read(out/'RECIPE_RANKING.json')
            self.assertTrue(result['planning_improved'])
            self.assertEqual(result['top_recipes'],['LOWER_CLEARANCE'])
            self.assertAlmostEqual(result['ranked'][0]['average_net_gain'],2.)

    def test_accepted_incidental_contact_is_not_a_failure_overlay(self):
        from tools.contact_coordination.physical_failure_evidence import failure_events
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);first=dict(robot_link='left_hand_thumb_0_link',control_frame=3)
            atomic_json(out/'PREGRASP_OBJECT_PROTECTION.json',dict(status='PASS',first_premature_contact=first))
            self.assertEqual(failure_events(out,dict(first_failure_latched=None),None,10),[])
            atomic_json(out/'PREGRASP_OBJECT_PROTECTION.json',dict(status='PREGRASP_OBJECT_CONTACT',first_premature_contact=first))
            self.assertEqual(failure_events(out,dict(first_failure_latched=None),None,10)[0]['causal_label'],'PREGRASP_OBJECT_CONTACT')

    def test_receipt_invalidates_changed_dependency_or_artifact(self):
        from tools.contact_coordination.run_train40_practical_calibration import signature,valid_receipt
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);artifact=out/'data';atomic_text(artifact,'original');deps=[dict(a=1)]
            p=out/'receipt.json';atomic_json(p,dict(status='PASS',dependency_hash=signature(deps),predecessor=None,artifacts=[record(artifact)]))
            self.assertTrue(valid_receipt(p,deps,None));self.assertFalse(valid_receipt(p,[dict(a=2)],None))
            atomic_text(artifact,'changed');self.assertFalse(valid_receipt(p,deps,None))

    def test_worker_exception_published_before_pool_drains(self):
        from concurrent.futures import Future
        from tools.contact_coordination.practical_planning import resolve_case
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);failed=Future();queued=Future();running=Future()
            failed.set_exception(AssertionError('Smoothing generated invalid edge'))
            running.set_running_or_notify_cancel()
            pending={failed:('recipe','source','method'),queued:('recipe','next','method'),running:('recipe','active','method')}
            with self.assertRaisesRegex(AssertionError,'invalid edge'):
                resolve_case(out,failed,pending,3,40,stage='paper_A_train40')
            progress=read(out/'WORK_PROGRESS.json');failure=read(progress['infrastructure_failure']['path'])
            self.assertTrue(queued.cancelled());self.assertFalse(running.cancelled())
            self.assertEqual(progress['cancelled_pending_jobs'],1)
            self.assertEqual(progress['stage'],'paper_A_train40')
            self.assertEqual(failure['source_id'],'source')
            self.assertEqual(failure['exception_type'],'AssertionError')
            self.assertEqual(failure['status'],'INFRASTRUCTURE_FAILURE')

    def test_backend_repair_requires_exact_baseline_evidence_and_regression(self):
        from tools.contact_coordination.practical_study import assert_backend
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);code=out/'backend.py';atomic_text(code,'original');before=record(code)
            manifest=dict(unchanged_backend=[before]);target=out/'ARCHITECTURE_BASELINE_MANIFEST.json'
            atomic_json(target,manifest);self.assertTrue(assert_backend(out)['backend_bytes_unchanged'])
            atomic_text(code,'repaired');after=record(code)
            with self.assertRaisesRegex(RuntimeError,'FROZEN_ARCHITECTURE_CHANGED'):assert_backend(out)
            evidence=out/'reproduction.json';atomic_json(evidence,dict(status='REPRODUCED'))
            regression=out/'regression.json';atomic_json(regression,dict(status='FAIL',repaired_backend=after))
            repair=dict(before=before,after=after,reproduction_evidence=[record(evidence)],regression=record(regression),
                reason='Unchecked merged edge',authorization='User permits reproducible software consistency bug repair')
            manifest['verified_software_consistency_repairs']=[repair];atomic_json(target,manifest)
            with self.assertRaisesRegex(RuntimeError,'INVALID_SOFTWARE_CONSISTENCY_REPAIR'):assert_backend(out)
            atomic_json(regression,dict(status='PASS',repaired_backend=after));repair['regression']=record(regression)
            atomic_json(target,manifest);self.assertFalse(assert_backend(out)['backend_bytes_unchanged'])
            atomic_text(code,'unapproved subsequent change')
            with self.assertRaisesRegex(RuntimeError,'FROZEN_ARCHITECTURE_CHANGED'):assert_backend(out)
            atomic_text(code,'repaired');atomic_json(evidence,dict(status='CHANGED'))
            with self.assertRaisesRegex(RuntimeError,'INVALID_SOFTWARE_CONSISTENCY_REPAIR'):assert_backend(out)

    def test_planning_failure_is_not_physics_failure_or_dropped_denominator(self):
        row=dict(source_id='fixture',paper_method='PAPER_B',internal_method='C_COUPLED',
            planning=dict(summary=dict(VALID_TARGET=False,first_causal_failure=dict(cause='NO_IK'),source_deviation_m2=None),planning_seconds=1.),
            physical=dict(complete_plan=False,physics_executed=False,first_failure=dict(cause='NO_IK',phase='APPROACH_CLEARANCE')))
        result=episode(row);self.assertFalse(result['PHYSICS_EXECUTED']);self.assertFalse(result['FULL_TASK']);self.assertEqual(result['first_failure'],'NO_IK')


if __name__=='__main__':unittest.main()
