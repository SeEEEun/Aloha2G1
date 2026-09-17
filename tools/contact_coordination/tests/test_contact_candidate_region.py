import unittest
import numpy as np
from tools.contact_coordination.contact_candidate_region import translation_bank


class ContactRegionTests(unittest.TestCase):
    def test_region_covers_object_axes_without_changing_calibrated_radius(self):
        offset=np.array([.0027819178066,.0009154079458,-.0006367166421])
        bank=translation_bank(offset);radius=np.linalg.norm(offset)
        self.assertEqual(len(bank),21)
        for proposal in bank:self.assertLessEqual(np.linalg.norm(proposal['offset']),radius+1e-15)
        for index,fraction in enumerate((0.,.5,1.)):
            np.testing.assert_array_equal(bank[index]['offset'],fraction*offset)
        for axis in np.eye(3):
            for sign in (-1,1):
                self.assertTrue(any(np.allclose(v['offset'],sign*radius*axis,atol=1e-12,rtol=0) for v in bank))

    def test_same_inputs_deterministic_and_changed_extent_propagates(self):
        offset=np.array([.001,.002,.001]);first=translation_bank(offset)
        second=translation_bank(offset);changed=translation_bank(2*offset)
        for a,b,c in zip(first,second,changed):
            np.testing.assert_array_equal(a['offset'],b['offset'])
            np.testing.assert_allclose(2*a['offset'],c['offset'],atol=1e-15,rtol=0)


if __name__=='__main__':unittest.main()
