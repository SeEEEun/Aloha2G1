import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .receiving_relation import build_region
from .scientific_cache import digest


class ReceivingRelationTests(unittest.TestCase):
    def setUp(self):
        source=np.eye(4);source[:3,:3]=Rotation.from_euler('z',.4).as_matrix();source[:3,3]=[.01,-.02,.005]
        self.phase=dict(source_id='SYNTHETIC_TRAIN_INPUT',hand_roles=dict(giver='left',receiver='right'),
            source_functional_tool_object_relations=dict(left=np.eye(4).tolist(),right=source.tolist()),
            relation_status=dict(right='INFERRED_FROM_LEFT_RIGID_CARRY'))
        self.contact=dict(side='right',T_HO=np.eye(4).tolist())

    def region(self,phase):
        return build_region(phase,self.contact,[.1175,.0725,.0775],[.003,0,0],[0,0,1])

    def test_right_relation_changes_target_not_left(self):
        changed=copy.deepcopy(self.phase);changed['source_functional_tool_object_relations']['right'][0][3]+=.002
        a,b=self.region(self.phase),self.region(changed)
        self.assertGreater(np.max(np.abs(a['target_T_H_O']-b['target_T_H_O'])),1e-6)
        self.assertEqual(self.phase['source_functional_tool_object_relations']['left'],changed['source_functional_tool_object_relations']['left'])
        np.testing.assert_array_equal(a['nominal_target_T_O_H'],b['nominal_target_T_O_H'])

    def test_source_axis_semantics_and_bounds(self):
        a=self.region(self.phase);changed=copy.deepcopy(self.phase)
        h=np.linalg.inv(changed['source_functional_tool_object_relations']['right'])
        h[:3,:3]=Rotation.from_euler('z',.03).as_matrix()@h[:3,:3]
        changed['source_functional_tool_object_relations']['right']=np.linalg.inv(h).tolist()
        b=self.region(changed)
        self.assertNotAlmostEqual(a['closing_line_rotation_rad'],b['closing_line_rotation_rad'],places=8)
        self.assertLess(np.linalg.norm(b['translation_object_m']),b['translation_radius_m'])
        self.assertLessEqual(abs(b['closing_line_rotation_rad']),b['closing_line_rotation_bound_rad'])

    def test_same_relation_reproduces_without_source_id_branch(self):
        a=self.region(self.phase);changed=copy.deepcopy(self.phase);changed['source_id']='ANOTHER_SOURCE_ID'
        b=self.region(changed)
        np.testing.assert_array_equal(a['target_T_H_O'],b['target_T_H_O'])

    def test_semantic_hash_ignores_plot_metadata_only(self):
        a=dict(source=self.phase,model_hash='abc',candidate_fractions=[0,.5,1])
        b=copy.deepcopy(a);b['plotting']=dict(dpi=900);b['source']['report_metadata']='new caption'
        self.assertEqual(digest(a),digest(b))
        b['source']['source_functional_tool_object_relations']['right'][0][3]+=.001
        self.assertNotEqual(digest(a),digest(b))
        self.assertNotEqual(digest(a),digest(dict(a,model_hash='changed')))


if __name__=='__main__':unittest.main()
