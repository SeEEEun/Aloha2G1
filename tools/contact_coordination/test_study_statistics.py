import unittest
from .study_statistics import paired_result,summarize_policy


class StatisticsTests(unittest.TestCase):
    def test_missing_policy_runs_are_not_zero_percent_success(self):
        r=summarize_policy([],['s'+str(i) for i in range(35)])
        for c in r['conditions'].values():
            self.assertEqual(c['primary_status'],'NOT_MEASURED');self.assertIsNone(c['full_task_rate']);self.assertEqual(c['not_attempted'],35)
        self.assertIsNone(r['paired']['difference_pp'])

    def test_exact_paired_discordance_and_nondegenerate_extreme_interval(self):
        r=paired_result([False]*4,[True]*4)
        self.assertEqual(r['McNemar_exact_p'],.125);self.assertEqual(r['difference_pp'],100.)
        self.assertLess(r['paired_interval95_pp'][0],100.);self.assertEqual(r['paired_interval95_pp'][1],100.)
        same=paired_result([False]*35,[False]*35)
        self.assertEqual(same['McNemar_exact_p'],1.);self.assertLess(same['paired_interval95_pp'][0],0.);self.assertGreater(same['paired_interval95_pp'][1],0.)

    def test_policy_abort_is_resolved_failure_but_infrastructure_is_unknown(self):
        ids=['s'+str(i) for i in range(35)]
        rows=[dict(source_id=ids[0],condition='A',terminal='POLICY_SAFETY_ABORT',unknown=False,full_task_success=False),
              dict(source_id=ids[0],condition='B',terminal='INFRASTRUCTURE_INVALID',unknown=True,full_task_success=None)]
        r=summarize_policy(rows,ids)
        self.assertEqual(r['conditions']['A']['resolved'],1);self.assertEqual(r['conditions']['B']['resolved'],0)
        self.assertEqual(r['conditions']['A']['policy_safety_aborts'],1);self.assertEqual(r['conditions']['B']['infrastructure_unknown'],1)


if __name__=='__main__':unittest.main()
