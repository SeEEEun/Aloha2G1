"""Small deterministic source-dependence, parity and numerical regressions."""
import copy
import inspect
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT, read
from .source_phase import pose
from .targets import make_goals, selection_costs
from .planner import quintic_retime, realize_phase_goals


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.config=read(ROOT/'configs/contact_coordination/hybrid_development_v1.json')['target']
        names=['APPROACH_START','LEFT_CLOSE_BEGIN','LEFT_GRASP_SOURCE','LEFT_LIFT_BEGIN',
               'LEFT_TRANSPORT_BEGIN','RIGHT_APPROACH_BEGIN','RIGHT_ACQUIRE_SOURCE',
               'LEFT_RELEASE_BEGIN','RIGHT_TRANSPORT_BEGIN','FINAL_RELEASE_BEGIN','TASK_END']
        self.phase=dict(source_id='SYNTHETIC',hand_roles={'giver':'left','receiver':'right'},
            events={n:{'time_s':float(t)} for n,t in zip(names,[0,1,2,3,4,5,6,8,9,10,11])},
            registered_wrist_object_relations={'left':pose(np.eye(3),[0,.1,0]),'right':pose(np.eye(3),[0,-.1,0])},
            initial_object_pose_world=pose(np.eye(3),[.2,0,.8]),approach_axis_world=[0,0,-1],
            relation_status={'left':'INFERRED','right':'INFERRED'},supported_grasp_patches='UNKNOWN')
        self.priors={'source_timestamp':np.arange(12.)}
        for side,y in [('left',-.1),('right',.1)]:
            self.priors[side+'_wrist_world']=np.asarray([pose(np.eye(3),[.2+.02*i,y,.8+.01*i]) for i in range(12)])

    def goals(self, phase=None, priors=None, enabled=True):
        return make_goals(phase or self.phase,priors or self.priors,self.config,enable_coupling=enabled)

    def test_reproducible_and_source_relation_used(self):
        first=self.goals();second=self.goals()
        np.testing.assert_array_equal(first['phase_goals'][1]['wrist_pose_world']['left'],second['phase_goals'][1]['wrist_pose_world']['left'])
        changed=copy.deepcopy(self.phase)
        changed['registered_wrist_object_relations']['left'][0,3]+=.013
        self.assertGreater(np.linalg.norm(self.goals(changed)['phase_goals'][1]['wrist_pose_world']['left']-
                                         first['phase_goals'][1]['wrist_pose_world']['left']),.01)

    def test_role_change_changes_active_hand(self):
        changed=copy.deepcopy(self.phase);changed['hand_roles']={'giver':'right','receiver':'left'}
        self.assertEqual(self.goals(changed)['phase_goals'][1]['active_hands'],['right'])
        self.assertEqual(self.goals()['phase_goals'][1]['active_hands'],['left'])

    def test_common_se3_equivariance_and_no_independent_rebasing(self):
        world=pose(Rotation.from_euler('xyz',[.2,-.3,.5]).as_matrix(),[1,-.7,.4])
        phase=copy.deepcopy(self.phase);priors=copy.deepcopy(self.priors)
        phase['initial_object_pose_world']=world@phase['initial_object_pose_world']
        phase['approach_axis_world']=world[:3,:3]@phase['approach_axis_world']
        for side in ('left','right'):
            priors[side+'_wrist_world']=world@priors[side+'_wrist_world']
        a,b=self.goals(),self.goals(phase,priors)
        for x,y in zip(a['phase_goals'],b['phase_goals']):
            for side in ('left','right'):
                np.testing.assert_allclose(world@x['wrist_pose_world'][side],y['wrist_pose_world'][side],atol=1e-12)
            np.testing.assert_allclose(np.linalg.inv(x['wrist_pose_world']['left'])@x['wrist_pose_world']['right'],
                                      np.linalg.inv(y['wrist_pose_world']['left'])@y['wrist_pose_world']['right'],atol=1e-12)

    def test_coupling_has_only_cross_factor_difference(self):
        a,b=self.goals(),self.goals(enabled=False)
        sa,sb=a['candidate_selection'],b['candidate_selection']
        self.assertEqual(sa['noncoupling_settings'],sb['noncoupling_settings'])
        np.testing.assert_array_equal(sa['candidate_object_offsets'],sb['candidate_object_offsets'])
        ar=sorted(sa['ranking'],key=lambda r:(r['left'],r['right']))
        br=sorted(sb['ranking'],key=lambda r:(r['left'],r['right']))
        for x,y in zip(ar,br):
            self.assertEqual(x['unary_cost'],y['unary_cost'])
            self.assertEqual(x['cross_hand_cost'],y['cross_hand_cost'])
            self.assertAlmostEqual(x['total_cost']-y['total_cost'],self.config['coupling_weight']*x['cross_hand_cost'])

    def test_cross_factor_effect_and_singleton_redundancy(self):
        a=pose(np.eye(3),[0,0,0]);b=pose(np.eye(3),[.1,0,0])
        bank={'left':[0,1],'right':[0,1]};pred={'left':[[a]*3,[b]*3],'right':[[a]*3,[b]*3]}
        unary={'left':[0.,1.],'right':[1.,0.]}
        off=selection_costs(bank,unary,pred,self.config,False)[0]
        on=selection_costs(bank,unary,pred,self.config,True)[0]
        self.assertEqual((off['left'],off['right']),(0,1))
        self.assertEqual(on['left'],on['right'])
        single={'left':[0],'right':[0]};pred={'left':[[a]*3],'right':[[a]*3]};unary={'left':[1.],'right':[2.]}
        self.assertEqual(selection_costs(single,unary,pred,self.config,True),selection_costs(single,unary,pred,self.config,False))

    def test_method_blind_backend_interface(self):
        self.assertEqual(list(inspect.signature(realize_phase_goals).parameters),['g1','goals','q0','config','candidate_validator','seed_postures'])
        for token in ('source_id','representation','INTERACTION_OURS','WRIST_REFERENCE'):
            self.assertNotIn(token,inspect.getsource(realize_phase_goals))

    def test_retiming_bounds_and_continuity(self):
        knots=np.array([[0.,0.],[1.,-.3],[-.7,.2]])
        v=np.array([.23,.17]);a=np.array([.12,.2])
        q,d=quintic_retime(knots,v,a,120.)
        np.testing.assert_array_equal(q[0],knots[0]);np.testing.assert_allclose(q[-1],knots[-1])
        self.assertTrue(np.all(np.max(abs(np.diff(q,axis=0)*120),axis=0)<=v+1e-9))
        self.assertTrue(np.all(np.max(abs(np.diff(q,n=2,axis=0)*120**2),axis=0)<=a+1e-9))

    def test_real_named_g1_known_answer_se3(self):
        from .source_phase import COMMON
        from tools.doll_handoff_retargeting.common import load_common_config,load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
        common=load_common_config(COMMON);g1=G1Kinematics(common,load_scene(common))
        natural=np.asarray(read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json')['g1_14_arm_initial_q_rad'])
        known=natural.copy();known[[0,3,7,10]] += [.04,.06,-.03,.05]
        g1.assign(known);targets={}
        for side in ('left','right'):
            matrix=g1.wrist_pose(side)
            matrix[:3,3]=g1.model_to_world_position(matrix[:3,3])
            matrix[:3,:3]=g1.model_to_world_rotation(matrix[:3,:3]);targets[side]=matrix
        goal=dict(name='KNOWN_ANSWER',active_hands=['left','right'],wrist_pose_world=targets,
                  position_tolerance_m=1e-5,orientation_tolerance_rad=1e-4)
        cfg=read(ROOT/'configs/contact_coordination/hybrid_development_v1.json')['planner']
        result=realize_phase_goals(g1,[goal],natural,cfg)
        preferred=dict(goal,preferred_q=known)
        kept=realize_phase_goals(g1,[preferred],natural,cfg,lambda a,b,g: {'valid':True})
        np.testing.assert_array_equal(kept['q'][-1],known)
        self.assertEqual(kept['phases'][0]['selected_seed'],'preferred_phase_configuration')
        rejected=realize_phase_goals(g1,[preferred],natural,cfg,lambda a,b,g: {'valid':False})
        self.assertFalse(rejected['phases'][0]['admissible'])
        g1.assign(natural);actual=g1.wrist_pose('left')
        actual[:3,3]=g1.model_to_world_position(actual[:3,3]);actual[:3,:3]=g1.model_to_world_rotation(actual[:3,:3])
        turned=actual.copy();turned[:3,:3]=turned[:3,:3]@Rotation.from_euler('z',.2).as_matrix()
        relaxed=dict(name='FREE_OPEN_HAND',active_hands=['left'],wrist_pose_world={'left':turned},position_tolerance_m=1e-5,orientation_tolerance_rad=1e-4)
        bounded=dict(cfg,max_nfev_per_seed_per_goal=1)
        strict=realize_phase_goals(g1,[relaxed],natural,bounded,lambda a,b,g:{'valid':True})
        self.assertFalse(strict['phases'][0]['admissible'])
        relaxed['orientation_region']='SO3'
        free=realize_phase_goals(g1,[relaxed],natural,bounded,lambda a,b,g:{'valid':True})
        self.assertTrue(free['phases'][0]['admissible'])
        self.assertGreater(free['phases'][0]['selected_errors']['left']['orientation_rad'],.1)
        self.assertEqual(result['status'],'PHASE_IK_SATISFIED')
        self.assertFalse(result['execution_valid'])


if __name__=='__main__':
    unittest.main()
