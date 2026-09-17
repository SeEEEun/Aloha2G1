import unittest
import numpy as np
from tools.contact_coordination.interaction_candidates import acquisition_bank


class SourceIngressTests(unittest.TestCase):
    def bank(self,axis):
        def contact(x):
            relation=np.eye(4);relation[0,3]=-x
            return dict(side='left',T_HO=relation,T_wrist_H=np.eye(4))
        phase=dict(source_id='fixture',hand_roles=dict(giver='left',receiver='right'),
            source_functional_tool_object_relations=dict(left=np.eye(4)),
            initial_object_pose_world=np.eye(4),approach_axis_world=axis)
        cal=dict(contacts=dict(pregrasp=contact(0),acquisition_intent=contact(.018)))
        return acquisition_bank(phase,cal,[.08,.06,.09])

    def test_source_direction_moves_pregrasp_with_fixed_contact_goal(self):
        a=self.bank([0,1,0]);b=self.bank([0,-1,0])
        for left,right in zip(a,b):
            np.testing.assert_array_equal(left['task_space_target'],right['task_space_target'])
            self.assertLess(left['targets']['PREGRASP'][1,3],right['targets']['PREGRASP'][1,3])
            info=left['acquisition_direction']
            self.assertLessEqual(np.linalg.norm(info['pregrasp_lateral_offset_world_m']),info['maximum_lateral_freedom_m']+1e-12)

    def test_opposite_source_approach_cannot_reverse_grasp_side(self):
        for candidate in self.bank([-1,0,0]):
            direction=candidate['acquisition_direction']['realized_prior_world']
            self.assertGreater(direction[0],.95)
        candidates=self.bank([0,1,0])
        self.assertEqual(len({c['task_space_target'].round(9).tobytes() for c in candidates}),5)


if __name__=='__main__':unittest.main()
