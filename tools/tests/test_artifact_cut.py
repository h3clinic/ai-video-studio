"""Pure artifact-editor validation tests; no files, media, encoders or network."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    'artifact_cut', Path(__file__).resolve().parents[1] / 'render-artifact-cut.py')
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


class ForegroundTests(unittest.TestCase):
    def test_white_background_becomes_fully_transparent_black(self):
        rgb = np.full((24, 24, 3), 255, dtype=np.uint8)
        rgba = renderer.opaque_foreground(rgb)
        self.assertEqual(rgba.shape, (24, 24, 4))
        self.assertEqual(rgba.dtype, np.uint8)
        self.assertTrue(np.all(rgba == 0))

    def test_near_white_compression_background_is_removed(self):
        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        rgb[:] = [242, 244, 246]
        self.assertTrue(np.all(renderer.opaque_foreground(rgb) == 0))

    def test_black_and_colored_foregrounds_remain_opaque(self):
        for color in ([0, 0, 0], [40, 100, 180], [240, 242, 255]):
            with self.subTest(color=color):
                rgb = np.zeros((24, 24, 3), dtype=np.uint8)
                rgb[:] = color
                rgba = renderer.opaque_foreground(rgb)
                np.testing.assert_array_equal(rgba[..., :3], rgb)
                self.assertTrue(np.all(rgba[..., 3] == 255))

    def test_enclosed_white_eye_is_preserved_not_globally_keyed(self):
        rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
        rgb[7:25, 7:25] = 0
        rgb[12:20, 12:20] = 255
        rgba = renderer.opaque_foreground(rgb)
        np.testing.assert_array_equal(rgba[16, 16], [255, 255, 255, 255])
        np.testing.assert_array_equal(rgba[9, 9], [0, 0, 0, 255])
        np.testing.assert_array_equal(rgba[0, 0], [0, 0, 0, 0])

    def test_border_connected_white_is_removed_even_inside_silhouette(self):
        rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
        rgb[7:25, 7:25] = 0
        rgb[12:20, 12:20] = 255
        rgb[:16, 14:18] = 255
        rgba = renderer.opaque_foreground(rgb)
        np.testing.assert_array_equal(rgba[16, 16], [0, 0, 0, 0])
        np.testing.assert_array_equal(rgba[20, 9], [0, 0, 0, 255])

    def test_soft_boundary_is_bounded_and_input_remains_unchanged(self):
        rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
        rgb[8:24, 8:24] = [32, 96, 160]
        original = rgb.copy()
        rgba = renderer.opaque_foreground(rgb)
        np.testing.assert_array_equal(rgb, original)
        self.assertTrue(np.any((rgba[..., 3] > 0) & (rgba[..., 3] < 255)))
        np.testing.assert_array_equal(rgba[16, 16], [32, 96, 160, 255])
        self.assertTrue(np.all(rgba[rgba[..., 3] == 0, :3] == 0))


class ShotValidationTests(unittest.TestCase):
    def test_exact_180_seconds_and_repeated_assets_are_allowed(self):
        for shots in ([{'asset': 'source-video', 'seconds': 180}],
                      [{'asset': 'agent-49', 'seconds': 30},
                       {'asset': 'history_r2', 'seconds': 120},
                       {'asset': 'agent-49', 'seconds': 30}]):
            with self.subTest(shots=shots):
                renderer.validate_shots(shots)

    def test_incomplete_and_overlong_timelines_are_rejected(self):
        for shots in ([], [{'asset': 'x', 'seconds': 179}],
                      [{'asset': 'x', 'seconds': 181}],
                      [{'asset': 'x', 'seconds': 90}, {'asset': 'y', 'seconds': 89}]):
            with self.subTest(shots=shots), self.assertRaises(ValueError):
                renderer.validate_shots(shots)

    def test_shot_duration_requires_positive_integer_not_boolean(self):
        for seconds in (None, True, False, 0, -1, 180.0, '180', float('nan'), float('inf')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                renderer.validate_shots([{'asset': 'x', 'seconds': seconds}])

    def test_missing_fields_and_unknown_fields_are_rejected(self):
        for shot in ({}, {'asset': 'x'}, {'seconds': 180},
                     {'asset': 'x', 'seconds': 180, 'crop': [0, 0, 10, 10]},
                     'x', ['x', 180], None):
            with self.subTest(shot=shot), self.assertRaises(ValueError):
                renderer.validate_shots([shot])

    def test_asset_ids_cannot_be_paths_urls_or_filter_expressions(self):
        for asset in ('', '.', '..', '../mix', 'dir/file', r'dir\file',
                      'https://example.com/media', 'a b', 'a;movie=x', 'agent-49.mov'):
            with self.subTest(asset=asset), self.assertRaises(ValueError):
                renderer.validate_shots([{'asset': asset, 'seconds': 180}])

    def test_non_string_asset_ids_are_rejected(self):
        for asset in (None, 49, True, [], {}):
            with self.subTest(asset=asset), self.assertRaises((TypeError, ValueError)):
                renderer.validate_shots([{'asset': asset, 'seconds': 180}])


if __name__ == '__main__':
    unittest.main()
