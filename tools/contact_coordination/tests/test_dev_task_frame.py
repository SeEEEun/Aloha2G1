"""Independent known-pose and gauge checks for the DEV scene adapter."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from tools.contact_coordination.dev_scene_manifest import normalize_registered_phase
from tools.contact_coordination.source_phase import pose


class DevTaskFrameTests(unittest.TestCase):
    def data(self):
        obj=pose(np.eye(3),[.2,.2,.8375])
        wrists={'left':np.asarray([pose(np.eye(3),[.1,.3,.8])]),
                'right':np.asarray([pose(Rotation.from_euler('z',.3).as_matrix(),[.5,.1,.9])])}
        phase=dict(initial_object_pose_world=obj,placement={'bin_center_xy_m':[.7,.1]},
                   approach_axis_world=[1.,0.,0.],hand_roles={'giver':'left','receiver':'right'},
                   source_functional_tool_object_relations={s:np.linalg.inv(v[0])@obj for s,v in wrists.items()})
        return phase,wrists,np.asarray([obj]),{'bin_center_task_xy_m':[.8,.2]}

    def test_known_pose_preserves_contacts_and_metric_geometry(self):
        p,w,o,m=self.data();saved=copy.deepcopy(p)
        result,rw,ro=normalize_registered_phase(p,w,o,m,SimpleNamespace(root_pose=np.eye(4)),
                         {'mode':'SOURCE_TASK_AXIS_TO_G1_BILATERAL_AXIS'})
        np.testing.assert_allclose(result['initial_object_pose_world'][:3,3],[.7,.7,.8375],atol=1e-12)
        np.testing.assert_allclose(result['approach_axis_world'],[0.,-1.,0.],atol=1e-12)
        self.assertEqual(result['hand_roles'],saved['hand_roles'])
        for s in w:
            np.testing.assert_allclose(np.linalg.inv(rw[s][0])@ro[0],np.linalg.inv(w[s][0])@o[0],atol=1e-12)
        np.testing.assert_array_equal(p['initial_object_pose_world'],saved['initial_object_pose_world'])

    def test_common_source_gauge_does_not_change_normalized_scene(self):
        p,w,o,m=self.data();g=SimpleNamespace(root_pose=np.eye(4));cfg={'mode':'SOURCE_TASK_AXIS_TO_G1_BILATERAL_AXIS'}
        a,aw,ao=normalize_registered_phase(p,w,o,m,g,cfg)
        change=pose(Rotation.from_euler('z',.41).as_matrix(),[.02,-.03,0.])
        p['initial_object_pose_world']=change@p['initial_object_pose_world']
        p['approach_axis_world']=change[:3,:3]@p['approach_axis_world']
        mb=(change@np.r_[m['bin_center_task_xy_m'],0.,1.])[:2]
        b,bw,bo=normalize_registered_phase(p,{s:change@v for s,v in w.items()},change@o,
                                          {'bin_center_task_xy_m':mb},g,cfg)
        np.testing.assert_allclose(a['initial_object_pose_world'],b['initial_object_pose_world'],atol=1e-12)
        for s in w:np.testing.assert_allclose(aw[s],bw[s],atol=1e-12)
        np.testing.assert_allclose(ao,bo,atol=1e-12)

    def test_unknown_or_degenerate_task_axis_rejected(self):
        p,w,o,m=self.data();g=SimpleNamespace(root_pose=np.eye(4))
        with self.assertRaises(ValueError):normalize_registered_phase(p,w,o,m,g,{'mode':'unknown'})
        with self.assertRaisesRegex(ValueError,'SOURCE_EVIDENCE_MISSING'):
            normalize_registered_phase(p,w,o,{'bin_center_task_xy_m':[.2,.2]},g,{'mode':'SOURCE_TASK_AXIS_TO_G1_BILATERAL_AXIS'})


if __name__=='__main__':unittest.main()
