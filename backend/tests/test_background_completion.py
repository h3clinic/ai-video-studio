import unittest
import numpy as np
from real_video.background_completion import preserve_known, behind_depth


class BackgroundCompletionTests(unittest.TestCase):
    def test_known_pixels_exact(self):
        a = np.zeros((4, 5, 3), dtype=np.uint8); b = np.full_like(a, 255)
        mask = np.zeros((4, 5), dtype=bool); mask[1:3, 2:4] = True
        result = preserve_known(a, b, mask)
        np.testing.assert_array_equal(result[~mask], a[~mask])
        np.testing.assert_array_equal(result[mask], b[mask])

    def test_depth_order(self):
        a = np.ones((3, 3)); b = a*2; mask = np.eye(3, dtype=bool)
        result = behind_depth(a, b, mask)
        self.assertTrue((result[mask] >= b[mask]+.15-1e-7).all())
        np.testing.assert_array_equal(result[~mask], a[~mask])

    def test_invalid_depth(self):
        with self.assertRaises(ValueError): behind_depth(np.array([np.nan]), np.ones(1), np.ones(1, dtype=bool))
