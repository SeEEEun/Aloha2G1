"""Outcome-independent collision-rule fixtures, including containment and holes."""
import ast
import unittest
from pathlib import Path
import numpy as np

from tools.common_geometry_confirmed_collision import (
    trimesh, surface_distance, classify_mesh_pair, solid_components,
)


class GeometryRuleTests(unittest.TestCase):
    tolerance=1e-5

    def boxes(self,offset):
        a=trimesh.creation.box(extents=[.1,.1,.1])
        b=a.copy();b.apply_translation(offset)
        return a,b

    def test_separated(self):
        a,b=self.boxes([.103,0,0])
        r=classify_mesh_pair(a,b,self.tolerance)
        self.assertEqual(r['classification'],'PROXY_ONLY_OVERLAP')
        self.assertAlmostEqual(r['detailed_separation_mm'],3.,places=8)

    def test_interpenetration(self):
        a,b=self.boxes([.08,.02,.01])
        self.assertEqual(classify_mesh_pair(a,b,self.tolerance)['classification'],'HARD_SELF_COLLISION')

    def test_containment_without_surface_crossing(self):
        a=trimesh.creation.box(extents=[.2,.2,.2]);b=trimesh.creation.box(extents=[.05,.05,.05])
        self.assertGreater(surface_distance(a,b)['separation_m'],0)
        self.assertEqual(classify_mesh_pair(a,b,self.tolerance)['classification'],'HARD_SELF_COLLISION')

    def test_contact_does_not_become_hard_without_depth(self):
        a,b=self.boxes([.1,0,0])
        self.assertEqual(classify_mesh_pair(a,b,self.tolerance)['classification'],'UNRESOLVED_GEOMETRY')

    def test_open_mesh_fails_closed_on_enclosure_overlap(self):
        a,b=self.boxes([.08,.02,.01]);a.update_faces(np.arange(len(a.faces)-2))
        self.assertFalse(a.is_watertight)
        self.assertEqual(classify_mesh_pair(a,b,self.tolerance)['classification'],'UNRESOLVED_GEOMETRY')

    def test_open_mesh_can_certify_enclosure_separation(self):
        a,b=self.boxes([.103,0,0]);a.update_faces(np.arange(len(a.faces)-2))
        self.assertEqual(classify_mesh_pair(a,b,self.tolerance)['classification'],'PROXY_ONLY_OVERLAP')

    def test_symmetry(self):
        a,b=self.boxes([.08,.02,.01])
        ra=classify_mesh_pair(a,b,self.tolerance);rb=classify_mesh_pair(b,a,self.tolerance)
        self.assertEqual(ra['classification'],rb['classification'])
        self.assertAlmostEqual(ra['detailed_separation_mm'],rb['detailed_separation_mm'])

    def test_no_method_or_side_identifiers(self):
        path=Path(__file__).resolve().parents[1]/'tools/common_geometry_confirmed_collision.py'
        tree=ast.parse(path.read_text())
        forbidden={'representation_mode','method','episode','task_success','physical_outcome','object_pose'}
        self.assertFalse({n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}&forbidden)
        strings={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertFalse(strings&{'WRIST','INTERACTION','left_shoulder_roll_link','torso_link'})


if __name__=='__main__':unittest.main()
