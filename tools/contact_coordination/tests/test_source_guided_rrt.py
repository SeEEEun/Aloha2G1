import unittest
import numpy as np
from tools.contact_coordination.joint_path_planner import plan,Validator,Budget,validate_path
from tools.contact_coordination.path_quality import evaluate,simplify


class Guide:
    def __init__(self,height):
        self.height=height;self.proposals=[np.array([x,height*np.sin(np.pi*(x+.8)/1.6)]) for x in (-.48,-.16,.16,.48)]
    def features(self,q):return dict(wrists=np.array([[q[0],q[1],0.]]))
    def reference(self,u):return np.array([[[-.8+1.6*t,self.height*np.sin(np.pi*t),0.]] for t in u])


class SourceGuidedTests(unittest.TestCase):
    def test_1_every_free_call_expands_even_when_direct_valid(self):
        r=plan([-.8,0],[.8,0],[-1,-1],[1,1],lambda q:True)
        self.assertTrue(r['rrt_api_called']);self.assertGreater(r['rrt_search_expanded'],0)
        self.assertFalse(r['direct_early_bypass']);self.assertGreaterEqual(r['global_samples'],1)

    def test_2_straight_best_in_trivial_environment(self):
        r=plan([-.8,0],[.8,0],[-1,-1],[1,1],lambda q:True,guide=Guide(0))
        self.assertTrue(r['straight_path_selected']);self.assertTrue(r['full_geometry_revalidated'])

    def test_3_obstacle_and_fixed_budget(self):
        valid=lambda q:not(abs(q[0])<.23 and abs(q[1])<.5)
        r=plan([-.8,0],[.8,0],[-1,-1],[1,1],valid,guide=Guide(.7))
        self.assertFalse(r['direct_path_valid']);self.assertEqual(r['status'],'PATH_FOUND')
        self.assertTrue(validate_path(r['path'],[-1,-1],[1,1],valid)['valid'])
        self.assertLessEqual(r['state_checks'],r['budget']['state_checks'])

    def test_4_quality_ranks_distinct_paths(self):
        a=np.array([[-.8,0],[.8,0]]);b=np.array([[-.8,0],[-.3,.7],[.2,-.6],[.8,0]])
        cost=lambda p:evaluate(p,[-1,-1],[1,1])['score']
        self.assertLess(cost(a),cost(b))
        r=plan(a[0],a[-1],[-1,-1],[1,1],lambda q:True)
        self.assertGreater(len(r['path_candidates']),1)
        selected=next(x for x in r['path_candidates'] if x['path_id']==r['selected_path_id'])
        self.assertEqual(selected['score'],min(x['score'] for x in r['path_candidates']))

    def test_5_soft_guidance_changes_preference_and_can_leave_prior(self):
        results=[plan([-.8,0],[.8,0],[-1,-1],[1,1],lambda q:True,guide=Guide(h)) for h in (.55,-.55)]
        self.assertTrue(all(r['source_guided_samples']>0 and r['global_samples']>0 for r in results))
        self.assertGreater(np.mean(results[0]['path'][:,1]),0.)
        self.assertLess(np.mean(results[1]['path'][:,1]),0.)
        bad=plan([-.8,0],[.8,0],[-1,-1],[1,1],lambda q:abs(q[1])<.15,guide=Guide(.8))
        self.assertEqual(bad['status'],'PATH_FOUND')

    def test_6_protected_object_swept_interior(self):
        checker=Validator([-1,-1],[1,1],lambda q:np.linalg.norm(q)>.2)
        self.assertTrue(checker.state([-.8,0]));self.assertTrue(checker.state([.8,0]))
        self.assertFalse(checker.edge([-.8,0],[.8,0]))

    def test_7_carried_object(self):
        robot=lambda q:q[1]>.1
        held=lambda q:robot(q) and not(-.2<q[0]<.2 and q[1]-.25<0.)
        a=[-.8,.2];b=[.8,.2]
        self.assertTrue(Validator([-1,-1],[1,1],robot).edge(a,b))
        self.assertFalse(Validator([-1,-1],[1,1],held).edge(a,b))

    def test_8_smoothing_revalidates(self):
        q=np.array([[-.8,0],[-.4,.3],[0,.7],[.4,.3],[.8,0]])
        check=Validator([-1,-1],[1,1],lambda q:True)
        score=lambda p:evaluate(p,[-1,-1],[1,1])
        new,evidence=simplify(q,check,score,np.random.default_rng(1729))
        self.assertLess(score(new)['joint_path_length_rad'],score(q)['joint_path_length_rad'])
        self.assertTrue(evidence['full_geometry_revalidated'])
        np.testing.assert_array_equal(new[[0,-1]],q[[0,-1]])

    def test_9_no_method_or_episode_objective_input(self):
        import inspect
        for function in (plan,evaluate):
            names=set(inspect.signature(function).parameters)
            self.assertFalse(names&{'method','source_id','golden','physical_success'})
        a=plan([-.8,0],[.8,0],[-1,-1],[1,1],lambda q:True,guide=Guide(.3))
        b=plan([-.8,0],[.8,0],[-1,-1],[1,1],lambda q:True,guide=Guide(.3))
        np.testing.assert_array_equal(a['path'],b['path'])

    def test_10_collinear_merge_checks_new_edge_samples_before_retention(self):
        # This narrow fixture isolates differing subdivision grids, without
        # claiming exact continuous collision detection below the resolution.
        blocked=.5*(77/128)
        valid=lambda q:abs(q[0]-blocked)>1e-5
        check=Validator([0.],[1.],valid)
        raw=np.array([[0.],[.18],[.36],[.5]])
        self.assertTrue(all(check.edge(a,b) for a,b in zip(raw[:-1],raw[1:])))
        self.assertFalse(check.edge(raw[0],raw[-1]))
        guide=type('CompactGuide',(),{'proposals':[np.array([.18]),np.array([.36])]})()
        budget=Budget(iterations=48,improvement_iterations=4)
        result=plan([0.],[.5],[0.],[1.],valid,budget=budget,guide=guide)
        self.assertGreater(result['canonical_edge_rejections'],0)
        self.assertLessEqual(result['state_checks'],budget.state_checks)
        for candidate in result['path_candidates']:
            self.assertTrue(validate_path(candidate['waypoints'],[0.],[1.],valid)['valid'])
        if result['status']=='PATH_FOUND':
            self.assertTrue(validate_path(result['path'],[0.],[1.],valid)['valid'])
        else:self.assertEqual(result['status'],'NO_CONNECTING_PATH')


if __name__=='__main__':unittest.main()
