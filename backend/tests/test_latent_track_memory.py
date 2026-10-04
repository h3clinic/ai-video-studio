"""CPU mechanism tests; synthetic arrays test algebra, not video quality."""

import inspect
import math
import unittest
from unittest.mock import patch

import torch

from real_video.gaussian_latent_memory import GaussianLatentMemory
from real_video.latent_track_memory import (
    ALPHA_MIN, OPACITY, build_anchor_memory_condition, planar_gaussian_cache,
)
from real_video.wan_temporal_control import wan_temporal_average


DIGEST = 'a'*64


class LatentTrackMemoryTests(unittest.TestCase):
    def setUp(self):
        self.anchor = torch.full((2, 4, 5), 3.)
        self.tracks = torch.tensor([[[1., 1.], [3., 2.]]]).expand(5, -1, -1).clone()
        self.visible = torch.ones(5, 2)
        self.ids = torch.tensor([99, 7])

    def build(self, **changes):
        arguments = dict(anchor=self.anchor, tracks=self.tracks, visibility=self.visible,
                         asset_digest=DIGEST, ids=self.ids)
        arguments.update(changes)
        return build_anchor_memory_condition(**arguments)

    def test_constant_anchor_recall_equals_constant_times_coverage(self):
        condition, metrics = self.build()
        self.assertEqual(tuple(condition.shape), (1, 3, 2, 4, 5))
        torch.testing.assert_close(condition[:, :2], 3*condition[:, 2:].expand(-1, 2, -1, -1, -1))
        self.assertEqual(metrics['source_writes'], 1)
        self.assertEqual(metrics['source_updated_gaussians'], 2)
        self.assertFalse(metrics['measured_depth'])
        self.assertFalse(metrics['predicted_motion'])
        self.assertFalse(condition.requires_grad)

    def test_immutable_identity_and_inputs_and_permutation(self):
        self.anchor = torch.arange(40).float().reshape(2, 4, 5)
        originals = [tensor.clone() for tensor in (self.anchor, self.tracks, self.visible, self.ids)]
        result, _ = self.build()
        permuted, _ = self.build(tracks=self.tracks[:, [1, 0]], visibility=self.visible[:, [1, 0]], ids=self.ids[[1, 0]])
        torch.testing.assert_close(result, permuted)
        for old, new in zip(originals, (self.anchor, self.tracks, self.visible, self.ids)):
            self.assertTrue(torch.equal(old, new))

    def test_numeric_id_sort_and_original_row_cache_keys(self):
        cache = planar_gaussian_cache(torch.zeros(2, 2), torch.ones(2), self.ids, 1, 1)
        self.assertEqual(cache['ids'].tolist(), [1, 0])
        self.assertEqual(cache['pixel'].tolist(), [0, 0])
        torch.testing.assert_close(cache['weight'], torch.tensor([.99, .0099]))

    def test_actual_gaussian_density_transmittance_and_truncation(self):
        cache = planar_gaussian_cache(torch.tensor([[2., 2.]]), torch.ones(1), torch.tensor([12]), 5, 5)
        weights = dict(zip(cache['pixel'].tolist(), cache['weight'].tolist()))
        self.assertAlmostEqual(weights[12], OPACITY, places=6)
        self.assertAlmostEqual(weights[13], OPACITY*math.exp(-1/(2*.35**2)), places=6)
        self.assertNotIn(14, weights)
        self.assertTrue(all(weight >= ALPHA_MIN for weight in weights.values()))

    def test_fractional_coordinates_are_not_pixel_edges_or_rounded(self):
        cache = planar_gaussian_cache(torch.tensor([[1.5, 1.]]), torch.ones(1), torch.tensor([0]), 3, 4)
        weights = dict(zip(cache['pixel'].tolist(), cache['weight'].tolist()))
        self.assertAlmostEqual(weights[5], weights[6], places=7)
        self.assertAlmostEqual(weights[5], OPACITY*math.exp(-.25/(2*.35**2)), places=6)

    def test_arbitrarily_narrow_positive_sigma_keeps_exact_center_finite(self):
        cache = planar_gaussian_cache(torch.tensor([[1., 1.]]), torch.ones(1), torch.tensor([0]), 3, 3, sigma=1e-200)
        self.assertEqual(cache['pixel'].tolist(), [4])
        self.assertAlmostEqual(float(cache['weight'][0]), OPACITY, places=6)

    def test_segmented_compositing_matches_independent_pixel_products(self):
        points = torch.tensor([[1.2, 1.], [.5, .8], [1., 1.], [2.5, 1.3]])
        visible = torch.tensor([1., .4, .8, .2])
        ids = torch.tensor([99, -1, 6, 200])
        cache = planar_gaussian_cache(points, visible, ids, 3, 4, sigma=.7)
        expected = {}
        for y in range(3):
            for x in range(4):
                transmittance = 1.
                for row in torch.argsort(ids).tolist():
                    distance2 = ((points[row].double()-torch.tensor([x, y])).square().sum()).item()
                    alpha = OPACITY*float(visible[row])*math.exp(-distance2/(2*.7**2))
                    if alpha >= ALPHA_MIN:
                        expected[(y*4+x, row)] = alpha*transmittance
                        transmittance *= 1-alpha
        self.assertEqual(len(expected), len(cache['weight']))
        for pixel, row, weight in zip(cache['pixel'].tolist(), cache['ids'].tolist(), cache['weight'].tolist()):
            self.assertAlmostEqual(weight, expected[(pixel, row)], places=6)

    def test_source_invisibility_never_acquires_future_appearance(self):
        visible = self.visible.clone()
        visible[0] = 0
        condition, metrics = self.build(visibility=visible)
        self.assertEqual(int(condition.count_nonzero()), 0)
        self.assertEqual(metrics['source_updated_gaussians'], 0)
        self.assertEqual(metrics['source_writes'], 1)

    def test_future_invisibility_and_outside_centers_are_empty_not_clipped(self):
        tracks = self.tracks.clone()
        tracks[1:, :, 0] = 5.01
        condition, _ = self.build(tracks=tracks)
        self.assertEqual(int(condition[:, :, 1:].count_nonzero()), 0)
        visibility = self.visible.clone()
        visibility[1:] = 0
        hidden, _ = self.build(visibility=visibility)
        self.assertEqual(int(hidden[:, :, 1:].count_nonzero()), 0)

    def test_outside_source_cannot_become_known_later(self):
        tracks = self.tracks.clone()
        tracks[0, :, 0] = -.01
        condition, metrics = self.build(tracks=tracks)
        self.assertEqual(int(condition.count_nonzero()), 0)
        self.assertEqual(metrics['source_updated_gaussians'], 0)

    def test_dense_collisions_finite_and_coverage_bounded(self):
        count = 1000
        cache = planar_gaussian_cache(torch.ones(count, 2), torch.ones(count), torch.arange(count), 3, 3)
        coverage = torch.zeros(9).index_add_(0, cache['pixel'], cache['weight'])
        self.assertTrue(bool(torch.isfinite(cache['weight']).all()))
        self.assertGreater(float(coverage[4]), .999)
        self.assertLessEqual(float(coverage.max()), 1.000001)
        self.assertGreaterEqual(float(cache['weight'].min()), 0.)

    def test_anchor_only_api_no_future_rgb_and_exactly_one_real_memory_write(self):
        signature = inspect.signature(build_anchor_memory_condition)
        self.assertEqual(set(signature.parameters), {'anchor', 'tracks', 'visibility', 'asset_digest', 'sigma', 'ids', 'return_snapshot', 'track_generations'})
        writes, reads = [], []
        original_write, original_read = GaussianLatentMemory.write, GaussianLatentMemory.read

        def write(memory, latent, cache, **kwargs):
            writes.append((latent.clone(), kwargs['confidence'].clone()))
            return original_write(memory, latent, cache, **kwargs)

        def read(memory, cache, height, width, **kwargs):
            reads.append(memory.writes)
            return original_read(memory, cache, height, width, **kwargs)

        with patch.object(GaussianLatentMemory, 'write', write), patch.object(GaussianLatentMemory, 'read', read):
            self.build()
        self.assertEqual(len(writes), 1)
        self.assertTrue(torch.equal(writes[0][0], self.anchor))
        self.assertTrue(torch.equal(writes[0][1], torch.ones(4, 5)))
        self.assertEqual(reads, [1]*5)

    def test_later_controls_do_not_change_earlier_condition_slots(self):
        tracks = self.tracks[:1].expand(9, -1, -1).clone()
        visible = torch.ones(9, 2)
        old, _ = self.build(tracks=tracks, visibility=visible)
        tracks[5:, :, 0] += 1
        new, _ = self.build(tracks=tracks, visibility=visible)
        self.assertTrue(torch.equal(old[:, :, :2], new[:, :, :2]))
        self.assertFalse(torch.equal(old[:, :, 2], new[:, :, 2]))

    def test_temporal_average_preserves_zero_and_uses_each_future_frame(self):
        visible = self.visible.clone()
        visible[2:] = 0
        condition, _ = self.build(visibility=visible)
        torch.testing.assert_close(condition[:, :, 1], condition[:, :, 0]/4)
        initial, _ = self.build(tracks=self.tracks[:1], visibility=self.visible[:1])
        self.assertTrue(torch.equal(condition[:, :, :1], initial))

    def test_memory_bytes_do_not_conflate_dense_controls_with_resident_bank(self):
        _, short = self.build()
        _, long = self.build(tracks=self.tracks[:1].expand(9, -1, -1), visibility=torch.ones(9, 2))
        self.assertEqual(short['memory'], long['memory'])
        self.assertGreater(long['condition_tensor_bytes'], short['condition_tensor_bytes'])
        self.assertGreater(long['supplied_track_tensor_bytes'], short['supplied_track_tensor_bytes'])
        self.assertEqual(short['memory']['frame_history_bytes'], 0)

    def test_default_ids_and_boolean_visibility(self):
        actual, _ = self.build(ids=None, visibility=self.visible.bool())
        expected, _ = self.build(ids=torch.arange(2))
        self.assertTrue(torch.equal(actual, expected))

    def test_saved_canonical_bank_restores_exact_recall_without_vae_or_anchor(self):
        self.anchor = torch.arange(40).float().reshape(2, 4, 5)
        self.tracks[1:, :, 0] -= .25
        self.visible[0, 1] = .6
        expected, metrics, bank = self.build(return_snapshot=True)
        restored = GaussianLatentMemory(bank['ids'], bank['channels'], bank['source_image_digest'])
        restored.restore(bank)
        self.assertEqual(restored.writes, 1)
        height, width = bank['latent_height'], bank['latent_width']
        source = bank['canonical_first_centers']
        inside = ((source[:, 0] >= 0) & (source[:, 0] <= width-1)
                  & (source[:, 1] >= 0) & (source[:, 1] <= height-1))
        source_visibility = bank['canonical_visibility']*inside*(bank['observation_mass'] > 0)
        recalled = []
        for time in range(5):
            points = source if time == 0 else self.tracks[time]
            visible = bank['canonical_visibility'] if time == 0 else self.visible[time]*source_visibility
            cache = planar_gaussian_cache(points, visible, bank['ids'], height, width, sigma=bank['sigma'])
            read = restored.read(cache, height, width, ids=bank['ids'])
            recalled.append(torch.cat((read['features'], read['coverage'][None])))
        actual = wan_temporal_average(torch.stack(recalled, dim=1)[None])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(bank['source_image_digest'], DIGEST)
        self.assertEqual(bank['coordinate_convention'], 'planar latentpixelcenter indices')
        self.assertEqual(metrics['source_writes'], 1)

    def test_snapshot_contains_no_future_history_and_is_not_aliased(self):
        _, _, short = self.build(return_snapshot=True)
        tracks = self.tracks[:1].expand(9, -1, -1).clone()
        tracks[1:, :, 0] += .5
        visible = torch.ones(9, 2)
        visible[1:] = .3
        _, _, long = self.build(return_snapshot=True, tracks=tracks, visibility=visible)
        self.assertEqual(short.keys(), long.keys())
        self.assertEqual(set(short), {
            'schema', 'asset_digest', 'channels', 'max_observation_mass', 'writes',
            'ids', 'features', 'observation_mass', 'condition_snapshot_schema',
            'canonical_first_centers', 'canonical_visibility', 'sigma', 'opacity',
            'alpha_min', 'latent_height', 'latent_width', 'coordinate_convention',
            'source_image_digest', 'collision_order',
        })
        for key, value in short.items():
            self.assertTrue(torch.equal(value, long[key]) if isinstance(value, torch.Tensor) else value == long[key])
        original_centers = short['canonical_first_centers'].clone()
        original_visibility = short['canonical_visibility'].clone()
        self.tracks.zero_()
        self.visible.zero_()
        self.assertTrue(torch.equal(short['canonical_first_centers'], original_centers))
        self.assertTrue(torch.equal(short['canonical_visibility'], original_visibility))
        short['ids'].zero_()
        self.assertTrue(torch.equal(self.ids, torch.tensor([99, 7])))

    def test_snapshot_option_preserves_default_two_return_contract(self):
        self.assertEqual(len(self.build()), 2)
        self.assertEqual(len(self.build(return_snapshot=False)), 2)
        self.assertEqual(len(self.build(return_snapshot=True)), 3)
        with self.assertRaises(ValueError):
            self.build(return_snapshot=1)

    def test_invalid_shapes_types_ranges_ids_and_digest_fail_closed(self):
        invalid = [
            dict(anchor=self.anchor[None]), dict(anchor=self.anchor.long()),
            dict(anchor=torch.empty(0, 4, 5)), dict(anchor=self.anchor*float('nan')),
            dict(tracks=self.tracks[:4]), dict(tracks=self.tracks[..., :1]),
            dict(tracks=self.tracks[:, :0]), dict(tracks=self.tracks.long()),
            dict(tracks=self.tracks*float('inf')), dict(visibility=self.visible[None]),
            dict(visibility=self.visible.long()), dict(visibility=self.visible*2),
            dict(visibility=self.visible*-1), dict(visibility=self.visible*float('nan')),
            dict(ids=torch.tensor([0, 0])), dict(ids=torch.tensor([0])),
            dict(ids=torch.tensor([1., 2.])), dict(asset_digest='not-a-digest'),
            dict(sigma=0), dict(sigma=-1), dict(sigma=True), dict(sigma=float('inf')),
        ]
        for changes in invalid:
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(ValueError):
                    self.build(**changes)


if __name__ == '__main__':
    unittest.main()
