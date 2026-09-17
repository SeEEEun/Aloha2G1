"""A's primary wrist keyframes must not repeat a whole phase's soft curve."""
import unittest
from unittest.mock import patch
import numpy as np
from tools.contact_coordination.planner import realize_phase_goals


class Kinematics:
    arm_limits=np.tile([-2.,2.],(14,1))
    def assign(self,q):self.q=np.asarray(q)
    def wrist_pose(self,side):
        pose=np.eye(4);pose[:3,3]=self.q[:3] if side=='left' else self.q[7:10];return pose
    def model_to_world_position(self,p):return np.asarray(p)


def endpoint(g,goal,previous,*args):
    q=np.asarray(previous).copy();q[:3]=np.asarray(goal['wrist_pose_world']['left'])[:3,3]
    candidate=dict(q=q,admissible=True,goal_satisfied=True,cost=0.,seed_index=0,errors={})
    return dict(phases=[dict(phase=goal['name'],candidates=[candidate])],q=np.asarray([previous,q]))


class WristContractTests(unittest.TestCase):
    def test_rebuilt_departure_region_receives_source_guidance_before_endpoint_search(self):
        from tools.contact_coordination.full_task_plan import realize_source_region
        u=np.linspace(0,1,7)
        prior=dict(object_task_frame_world=np.eye(4),phase_u=u,
            wrist_anchors_object_frame=dict(left=np.c_[.2*u,.08*np.sin(np.pi*u),0*u]))
        target=np.eye(4);target[0,3]=.2
        # A geometry-generated candidate has no inherited source-motion field.
        goal=dict(name='RECEIVER_DEPARTURE',active_hands=['left'],wrist_pose_world=dict(left=target))
        with patch('tools.contact_coordination.source_motion_prior.extract',return_value=prior), \
             patch('tools.contact_coordination.planner.solve_endpoints',side_effect=endpoint):
            result=realize_source_region(Kinematics(),[goal],np.zeros(14),{},
                lambda *args:dict(valid=True,minimum_clearance_m=.03),source_folder='source_fixture')
        self.assertTrue(result['valid'])
        cert=result['attempts'][0]['result']['phases'][0]['planner_attempts'][0]
        self.assertGreater(cert['source_guided_samples'],0)
        self.assertGreater(cert['rrt_search_expanded'],0)
        np.testing.assert_array_equal(cert['source_guidance']['compact_prior']['phase_u'],u)

    def test_primary_wrist_segments_use_local_reference_not_repeated_phase_curve(self):
        u=np.linspace(0,1,7);prior=dict(object_task_frame_world=np.eye(4),phase_u=u,
            wrist_anchors_object_frame=dict(left=np.c_[.2*u,.08*np.sin(np.pi*u),0*u]))
        target=np.eye(4);target[0,3]=.2;middle=target.copy();middle[0,3]=.1
        goal=dict(name='LEFT_CARRY',active_hands=['left'],wrist_pose_world=dict(left=target),source_motion_prior=prior)
        a=dict(goal,source_wrist_reference_waypoints=[dict(wrist_pose_world=dict(left=middle))])
        validator=lambda *args:dict(valid=True,minimum_clearance_m=.03)
        with patch('tools.contact_coordination.planner.solve_endpoints',side_effect=endpoint):
            baseline=realize_phase_goals(Kinematics(),[a],np.zeros(14),{},validator)
            interaction=realize_phase_goals(Kinematics(),[goal],np.zeros(14),{},validator)
        phases=baseline['phases'][0]['source_wrist_reference']['phases']
        self.assertEqual(len(phases),2)
        for phase in phases:
            cert=phase['planner_attempts'][0]
            self.assertTrue(cert['rrt_api_called']);self.assertGreater(cert['rrt_search_expanded'],0)
            self.assertIsNone(cert['source_guidance']['compact_prior'])
        ours=interaction['phases'][0]['planner_attempts'][0]
        self.assertIsNotNone(ours['source_guidance']['compact_prior']);self.assertGreater(ours['source_guided_samples'],0)


if __name__=='__main__':unittest.main()
