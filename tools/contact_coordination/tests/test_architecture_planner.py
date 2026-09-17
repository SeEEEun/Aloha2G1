"""Nontrivial obstacle and midpoint regressions for the actual search module."""
import unittest
import numpy as np
from tools.contact_coordination.joint_path_planner import Budget,Validator,plan,validate_path


class PathSearchTests(unittest.TestCase):
    def test_obstacle_detour(self):
        def valid(q):return not (abs(q[0])<.23 and abs(q[1])<.5)
        a=np.array([-.8,0.]);b=np.array([.8,0.]);lo=np.full(2,-1.);hi=-lo
        checker=Validator(lo,hi,valid)
        self.assertTrue(checker.state(a));self.assertTrue(checker.state(b))
        self.assertFalse(checker.edge(a,b))
        result=plan(a,b,lo,hi,valid)
        self.assertEqual(result['status'],'PATH_FOUND',result)
        self.assertTrue(result['search_used']);self.assertFalse(result['direct_path_valid'])
        self.assertTrue(validate_path(result['path'],lo,hi,valid)['valid'])
        repeat=plan(a,b,lo,hi,valid)
        np.testing.assert_array_equal(result['path'],repeat['path'])

    def test_valid_endpoints_invalid_midpoint(self):
        checker=Validator([-1.],[1.],lambda q:abs(q[0])>.01)
        self.assertTrue(checker.state([-.8]));self.assertTrue(checker.state([.8]))
        self.assertFalse(checker.edge([-.8],[.8]))

    def test_no_path_fixed_budget(self):
        result=plan([-.8,0.],[.8,0.],[-1.,-1.],[1.,1.],lambda q:abs(q[0])>.1,
                    budget=Budget(iterations=40,state_checks=1200))
        self.assertEqual(result['status'],'NO_CONNECTING_PATH')
        self.assertLessEqual(result['state_checks'],1200)

    def test_carried_object_obstacle(self):
        # Robot reference point clears wall; its rigidly attached box hits it.
        robot=lambda q:not (.1<q[0]<.3 and .2<q[1]<.4)
        attached=lambda q:robot(q) and not (-.1<q[0]<.5 and -.01<q[1]<.61)
        lo=[-1.,-1.];hi=[1.,1.];a=[-.8,0.];b=[.8,0.]
        self.assertTrue(Validator(lo,hi,robot).edge(a,b))
        self.assertFalse(Validator(lo,hi,attached).edge(a,b))

    def test_feedback_contact_jump_reversal_and_joint_boundary_retiming(self):
        from tools.contact_coordination.contact_command_retiming import ContactCommandRetimer,contact_limits
        dt,v,a=contact_limits();v=v[5:6];a=a[5:6]
        # Captured left index latch: the old three-frame transition jumped
        # from -.19779733 to -.26117310 to -.32454888 radians.
        captured=np.r_[[-.18305537,-.18674086,-.19042635,-.19411184,-.19779733,
                        -.19779733,-.26117310,-.32454888],np.full(80,-.32454888)]
        targets=np.r_[captured,np.full(120,-.8),np.full(120,-.01),np.full(120,-.75)]
        retimer=ContactCommandRetimer(v,a,[-.8],[0.],dt)
        q=np.asarray([retimer.step([x]) for x in targets])
        dq=np.diff(q,axis=0)/dt;ddq=np.diff(dq,axis=0)/dt
        self.assertLessEqual(float(np.max(abs(dq)/v)),1.+1e-9)
        self.assertLessEqual(float(np.max(abs(ddq)/a)),1.+1e-9)
        self.assertTrue(np.all((q>=-.8-1e-12)&(q<=1e-12)))
        self.assertLess(float(abs(q[87,0]+.32454888)),1e-10)
        self.assertGreater(retimer.changed_frames,0)
        # The original already-admissible approach samples are retained.
        np.testing.assert_allclose(q[:6,0],captured[:6],atol=1e-12)

    def test_contact_retimer_handles_changing_targets_all_fingers(self):
        from tools.contact_coordination.contact_command_retiming import ContactCommandRetimer,contact_limits
        dt,v,a=contact_limits();rng=np.random.default_rng(1729)
        retimer=ContactCommandRetimer(v,a,np.full(14,-1.),np.full(14,1.),dt)
        targets=np.repeat(rng.uniform(-1.,1.,(100,14)),7,axis=0)
        q=np.array([retimer.step(x) for x in targets])
        velocity=np.diff(q,axis=0)/dt;acceleration=np.diff(velocity,axis=0)/dt
        self.assertLessEqual(float(np.max(abs(velocity)/v)),1.+1e-9)
        self.assertLessEqual(float(np.max(abs(acceleration)/a)),1.+1e-9)
        self.assertTrue(np.all(abs(q)<=1.+1e-12))


if __name__=='__main__':unittest.main()
