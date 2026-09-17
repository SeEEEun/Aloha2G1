"""Synthetic numerical unit tests, not physical task evidence."""
import unittest
import numpy as np
from .score_policy import measured_stages


class PolicyScoringTests(unittest.TestCase):
    def trace(self):
        n=1000;pos=np.tile([.3,.3,.8375],(n,1));pos[100:800,2]=1.02
        pos[600:801,:2]=np.linspace([.3,.3],[.738212049,.099787664],201);pos[801:,:2]=pos[800,:2]
        pos[800:,2]=.85
        a=dict(object_position_world_m=pos,object_linear_velocity_m_s=np.zeros((n,3)),control_frame=np.arange(n),
            table_contact_force_n=np.r_[np.ones(100),np.zeros(900)],doll_bin_contact_force_n=np.r_[np.zeros(800),np.ones(200)],
            EXECUTED_COMMAND=np.ones((n,28)),right_open_q=np.zeros(7))
        a['EXECUTED_COMMAND'][850:,21:]=0.
        for side,begin,end in [('left',0,450),('right',300,850)]:
            for digit in ('thumb','index','middle'):
                a[f'{side}_{digit}_force_n']=np.zeros(n);a[f'{side}_{digit}_force_n'][begin:end]=.1
            a[f'{side}_palm_force_n']=np.zeros(n)
        return a

    def test_full_contact_sequence_and_release_without_source_clock(self):
        a=self.trace();r=measured_stages(a,.01)
        self.assertTrue(r['stages']['FULL_TASK']);self.assertEqual(r['release_classification'],'POLICY_OPENING_RELEASE_IN_VALID_BIN_REGION')
        a['stage']=np.repeat('FAKE_NO_TASK',1000);a['source_events']=np.arange(1000)[::-1]
        self.assertEqual(measured_stages(a,.01),r)

    def test_dual_support_alone_is_not_handoff_or_right_ownership(self):
        a=self.trace()
        for d in ('thumb','index','middle'):a['left_'+d+'_force_n'][:]=.1
        r=measured_stages(a,.01)
        self.assertIn('DUAL_SUPPORT',r['event_rows']);self.assertFalse(r['stages']['HANDOFF']);self.assertFalse(r['stages']['RIGHT_OWNERSHIP'])

    def test_visual_bin_entry_without_acquisition_is_not_cumulative_completion(self):
        a=self.trace()
        for k in a:
            if k.startswith('left_'):a[k][:]=0.
        r=measured_stages(a,.01)
        self.assertTrue(r['raw_bin_entry_observed']);self.assertFalse(any(r['stages'].values()));self.assertEqual(r['release_classification'],'NO_ACQUISITION')

    def test_settle_requires_the_entire_last_second(self):
        a=self.trace();a['object_linear_velocity_m_s'][-100,0]=.020001
        self.assertFalse(measured_stages(a,.01)['stages']['FULL_TASK'])


if __name__=='__main__':unittest.main()
