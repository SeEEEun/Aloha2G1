"""Additional independent geometric properties; no episode data or outcomes."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from tools import common_geometry_confirmed_collision as geometry


class GeometryProperties(unittest.TestCase):
    tolerance=1e-5

    def pair(self,penetration):
        a=geometry.trimesh.creation.box(extents=[.1,.1,.1])
        b=a.copy();b.apply_translation([.1-penetration,.02,.01])
        return a,b

    def test_sub_tolerance_overlap_not_hard(self):
        a,b=self.pair(5e-6)
        self.assertEqual(geometry.classify_mesh_pair(a,b,self.tolerance)['classification'],'UNRESOLVED_GEOMETRY')

    def test_fifty_micrometer_overlap_is_hard(self):
        a,b=self.pair(50e-6)
        self.assertEqual(geometry.classify_mesh_pair(a,b,self.tolerance)['classification'],'HARD_SELF_COLLISION')

    def test_rigid_transform_invariance(self):
        a,b=self.pair(-.003)
        before=geometry.classify_mesh_pair(a,b,self.tolerance)
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',[.23,-.71,1.11]).as_matrix();t[:3,3]=[.31,-.54,.92]
        a.apply_transform(t);b.apply_transform(t)
        after=geometry.classify_mesh_pair(a,b,self.tolerance)
        self.assertEqual(before['classification'],after['classification'])
        self.assertAlmostEqual(before['detailed_separation_mm'],after['detailed_separation_mm'],places=8)

    def test_open_enclosure_containment_is_unresolved(self):
        a=geometry.trimesh.creation.box(extents=[.2,.2,.2]);a.update_faces(np.arange(10))
        b=geometry.trimesh.creation.box(extents=[.02,.02,.02])
        self.assertGreater(geometry.surface_distance(a,b)['separation_m'],self.tolerance)
        self.assertEqual(geometry.classify_mesh_pair(a,b,self.tolerance)['classification'],'UNRESOLVED_GEOMETRY')

    def test_crossing_triangle_edges(self):
        a=geometry.trimesh.Trimesh(vertices=[[-1,-1,0],[1,-1,0],[0,1,0]],faces=[[0,1,2]],process=False)
        b=geometry.trimesh.Trimesh(vertices=[[0,0,-1],[0,0,1],[0,2,0]],faces=[[0,1,2]],process=False)
        self.assertEqual(geometry.surface_distance(a,b)['separation_m'],0.)

    def test_finite_triangle_budget(self):
        a,b=self.pair(.01)
        with patch.object(geometry,'MAX_TRIANGLE_PAIRS',1):
            with self.assertRaisesRegex(RuntimeError,'budget exceeded'):geometry.surface_distance(a,b)

    def test_no_rng_consumption(self):
        a,b=self.pair(.01)
        state=np.random.get_state()
        geometry.classify_mesh_pair(a,b,self.tolerance)
        after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])


if __name__=='__main__':unittest.main()
