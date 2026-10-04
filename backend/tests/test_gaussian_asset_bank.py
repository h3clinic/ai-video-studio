"""Adversarial contract tests; stable point IDs are not semantic identities."""
import copy
import math
import unittest

import torch

from real_video.gaussian_asset_bank import (
    GaussianAssetBank, bank_from_points, geometry_hash, make_motion, validate_bank,
)


class GaussianAssetBankTests(unittest.TestCase):
    def setUp(self):
        self.points = torch.tensor([
            [.2, .3, -4., -3., 1., 0., .2, -.3, .4, .8],
            [.4, .5, -3., -4., 0., 1., -.1, .6, .8, 1.],
            [.6, .7, -4., -4., -1., 0., .5, .3, -.2, .4],
        ])
        self.ids = torch.tensor([90, 2, 31], dtype=torch.int64)
        self.bank = bank_from_points(self.points, ids=self.ids)
        self.asset = GaussianAssetBank(self.bank)
        self.delta = torch.tensor([[.01, .02], [-.03, .04], [.05, -.06]])

    def packet(self, **kwargs):
        return make_motion(self.bank, self.ids, self.delta, **kwargs)

    def test_zero_motion_exact_and_output_owned(self):
        packet = make_motion(self.bank, self.ids, torch.zeros_like(self.delta))
        output = self.asset.attach(packet)
        self.assertTrue(torch.equal(output, self.points))
        output.fill_(100)
        self.assertTrue(torch.equal(self.asset.attach(packet), self.points))

    def test_motion_row_shuffle_with_all_optional_fields(self):
        packet = self.packet(angle=torch.tensor([.2, -.3, .4]),
                             log_scale_delta=self.delta * 2,
                             visibility=torch.tensor([[.2], [.7], [1.]]))
        shuffled = copy.deepcopy(packet)
        permutation = torch.tensor([2, 0, 1])
        for key in ['ids', 'displacement', 'angle', 'log_scale_delta', 'visibility']:
            shuffled[key] = shuffled[key][permutation]
        self.assertTrue(torch.equal(self.asset.attach(packet), self.asset.attach(shuffled)))

    def test_bank_owns_source_points_and_ids(self):
        self.points.fill_(100)
        self.ids.fill_(100)
        validate_bank(self.bank)
        self.assertTrue(torch.equal(self.bank['ids'], torch.tensor([90, 2, 31])))
        self.assertTrue(torch.equal(self.asset.reference[:, :2], self.bank['geometry']['position']))

    def test_runtime_owns_bank_tensors(self):
        packet = self.packet()
        expected = self.asset.attach(packet)
        self.bank['geometry']['position'].fill_(100)
        self.bank['appearance']['colour'].fill_(100)
        self.bank['ids'].fill_(100)
        self.assertTrue(torch.equal(self.asset.attach(packet), expected))

    def test_packet_owns_input_tensors_and_attach_does_not_mutate(self):
        angle = torch.tensor([.1, .2, .3])
        scale = self.delta.clone()
        visibility = torch.tensor([[.1], [.2], [.3]])
        packet = self.packet(angle=angle, log_scale_delta=scale, visibility=visibility)
        saved = copy.deepcopy(packet)
        self.ids.fill_(7); self.delta.fill_(100)
        angle.fill_(100); scale.fill_(100); visibility.fill_(100)
        self.asset.attach(packet)
        for key in ['ids', 'displacement', 'angle', 'log_scale_delta', 'visibility']:
            self.assertTrue(torch.equal(packet[key], saved[key]), key)

    def test_wrong_geometry_bank_rejected(self):
        other = self.points.clone(); other[0, 0] += .01
        bank = bank_from_points(other, ids=self.ids)
        with self.assertRaisesRegex(ValueError, 'different geometry'):
            self.asset.attach(make_motion(bank, self.ids, self.delta))

    def test_duplicate_unknown_and_missing_motion_ids_rejected(self):
        for ids, delta in [(torch.tensor([90, 90, 31]), self.delta),
                           (torch.tensor([90, 2, 30]), self.delta),
                           (torch.tensor([90, 2, 999]), self.delta),
                           (self.ids[:2], self.delta[:2])]:
            with self.subTest(ids=ids.tolist()), self.assertRaises(ValueError):
                self.asset.attach(make_motion(self.bank, ids, delta))

    def test_explicit_partial_packet_leaves_unmentioned_points_exact(self):
        packet = make_motion(self.bank, self.ids[1:2], self.delta[1:2], partial=True)
        result = self.asset.attach(packet)
        expected = self.points.clone(); expected[1, :2] += self.delta[1]
        self.assertTrue(torch.equal(result, expected))
        packet['partial'] = False
        with self.assertRaises(ValueError): self.asset.attach(packet)

    def test_colour_and_order_only_cannot_render(self):
        incomplete = dict(version=1, ids=self.ids,
                          appearance=copy.deepcopy(self.bank['appearance']))
        with self.assertRaisesRegex(ValueError, 'colour/order alone'):
            GaussianAssetBank(incomplete)

    def test_rotation_preserves_unit_axes_and_other_attributes(self):
        angle = torch.tensor([math.pi / 2, math.pi, -.3])
        packet = make_motion(self.bank, self.ids, torch.zeros_like(self.delta), angle=angle)
        result = self.asset.attach(packet)
        self.assertTrue(torch.allclose(result[:, 4:6].norm(dim=-1), torch.ones(3), atol=1e-6))
        self.assertTrue(torch.allclose(result[0, 4:6], torch.tensor([0., 1.]), atol=1e-6))
        self.assertTrue(torch.equal(result[:, :4], self.points[:, :4]))
        self.assertTrue(torch.equal(result[:, 6:], self.points[:, 6:]))

    def test_nonfinite_bank_fields_rejected(self):
        for section in ['geometry', 'appearance']:
            for name in self.bank[section]:
                for invalid in [float('nan'), float('inf'), -float('inf')]:
                    bank = copy.deepcopy(self.bank)
                    bank[section][name][0, 0] = invalid
                    with self.subTest(section=section, name=name, invalid=invalid), self.assertRaises(ValueError):
                        GaussianAssetBank(bank)

    def test_nonfinite_motion_fields_rejected(self):
        for name, shape in [('displacement', (3, 2)), ('angle', (3,)),
                            ('log_scale_delta', (3, 2)), ('visibility', (3, 1))]:
            for invalid in [float('nan'), float('inf'), -float('inf')]:
                packet = self.packet()
                packet[name] = torch.zeros(shape)
                packet[name].flatten()[0] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    self.asset.attach(packet)

    def test_colour_edit_preserves_geometry_binding(self):
        changed = copy.deepcopy(self.bank)
        changed['appearance']['colour'] *= -.5
        self.assertEqual(geometry_hash(changed), self.bank['geometry_hash'])
        output = GaussianAssetBank(changed).attach(self.packet())
        self.assertTrue(torch.equal(output[:, :6], self.asset.attach(self.packet())[:, :6]))
        self.assertTrue(torch.equal(output[:, 6:9], changed['appearance']['colour']))

    def test_bank_row_permutation_hash_and_motion_association(self):
        order = torch.tensor([1, 2, 0])
        reordered = bank_from_points(self.points[order], ids=self.ids[order])
        self.assertEqual(reordered['geometry_hash'], self.bank['geometry_hash'])
        output = GaussianAssetBank(reordered).attach(self.packet())
        self.assertTrue(torch.equal(output, self.asset.attach(self.packet())[order]))

    def test_invalid_ids_and_geometry_integrity(self):
        for ids in [torch.tensor([90, 90, 31]), torch.tensor([-1, 2, 31]),
                    self.ids.float()]:
            bank = copy.deepcopy(self.bank); bank['ids'] = ids
            with self.subTest(ids=ids.tolist()), self.assertRaises(ValueError):
                validate_bank(bank)
        altered = copy.deepcopy(self.bank)
        altered['geometry']['position'][0, 0] += .1
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): validate_bank(altered)

    def test_visibility_range_and_motion_shape_rejected(self):
        for visibility in [torch.full((3, 1), -.1), torch.full((3, 1), 1.1)]:
            with self.assertRaises(ValueError): self.asset.attach(self.packet(visibility=visibility))
        packet = self.packet(); packet['displacement'] = torch.zeros(3, 3)
        with self.assertRaises(ValueError): self.asset.attach(packet)

    def test_constructor_rejects_malformed_ids_before_indexing(self):
        for ids in [self.ids[:2], self.ids[:, None], self.ids.float(), torch.tensor([1, 1, 2])]:
            with self.subTest(shape=ids.shape), self.assertRaises(ValueError):
                bank_from_points(self.points, ids=ids)

    def test_float_contract_and_consistent_bank_dtype(self):
        with self.assertRaises(ValueError): bank_from_points(self.points.int())
        bank = copy.deepcopy(self.bank)
        bank['geometry']['axis'] = bank['geometry']['axis'].double()
        with self.assertRaisesRegex(ValueError, 'matching dtype'): validate_bank(bank)
        packet = self.packet(); packet['displacement'] = self.delta.int()
        with self.assertRaises(ValueError): self.asset.attach(packet)

    def test_partial_flag_and_timestamp_metadata_are_validated(self):
        for value in ['false', 1, None]:
            with self.assertRaises(ValueError): self.packet(partial=value)
            packet = self.packet(); packet['partial'] = value
            with self.assertRaises(ValueError): self.asset.attach(packet)
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError): self.packet(time_seconds=value)
            packet = self.packet(); packet['time_seconds'] = value
            with self.assertRaises(ValueError): self.asset.attach(packet)


if __name__ == '__main__': unittest.main()
