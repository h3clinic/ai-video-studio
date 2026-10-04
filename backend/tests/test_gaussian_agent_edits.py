"""Tiny numerical diagnostics; these do not demonstrate generated video quality."""
from dataclasses import replace
import math
import unittest

from real_video.gaussian_agent_edits import (RigidTransform, Recolour, TexturePaint,
    TextureImage, TextureBinding, operation_from_dict, apply_agent_operations,
    rigid_motion_at, state_from_asset, state_to_arrays, canonical_state_from_npz)
from real_video.gaussian_part_ownership import PartBinding, OwnershipManifest


class GaussianAgentEditTests(unittest.TestCase):
    def setUp(self):
        self.state = {gid: dict(position=[float(gid), 0., 0.], colour=[.2, .4, .6], opacity=.8,
                               covariance=[[1., 0., 0.], [0., 4., 0.], [0., 0., 9.]]) for gid in (7, 11, 99)}
        self.part = PartBinding('fruit_skin', (11, 7), agent_id='material_worker', binding_verified=True)
        self.protected = PartBinding('actor', (99,), protected=True, binding_verified=True)
        self.manifest = OwnershipManifest(3, (self.part, self.protected))
        self.identity = 'a'*64

    def run_ops(self, *ops, **kwargs):
        return apply_agent_operations(self.state, self.manifest, ops, **kwargs)

    def test_rotation_covariance_and_protected_state(self):
        op = RigidTransform('material_worker', 'fruit_skin', 3,
                            ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)), (1., 2., 0.), (7., 0., 0.))
        result, manifest, receipts = self.run_ops(op)
        self.assertEqual(result[11]['position'], (8., 6., 0.))
        self.assertEqual(result[11]['covariance'], ((4., 0., 0.), (0., 1., 0.), (0., 0., 9.)))
        self.assertEqual(result[99]['position'], tuple(self.state[99]['position']))
        self.assertEqual(self.state[11]['position'], [11., 0., 0.])
        self.assertEqual(manifest.revision, 4)
        self.assertFalse(receipts[0]['learned_generation'])

    def test_recolour_and_transform_compose_one_atomic_revision(self):
        result, manifest, _ = self.run_ops(Recolour('material_worker', 'fruit_skin', 3, (1., 0., 0.), .5),
                                         RigidTransform('material_worker', 'fruit_skin', 3, translation=(1., 0., 0.)))
        self.assertEqual(result[7]['colour'], (.6, .2, .3))
        self.assertEqual(result[7]['position'], (8., 0., 0.))
        self.assertEqual(manifest.revision, 4)

    def test_texture_mapping_is_by_id_not_row_and_bilinear(self):
        image = TextureImage([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 1.]]], 'b'*64)
        binding = TextureBinding('fruit_skin', self.identity, {7: (0., 0.), 11: (.5, .5)}, True)
        op = TexturePaint('material_worker', 'fruit_skin', 3, 'skin')
        result, _, receipts = self.run_ops(op, textures={'skin': image}, texture_bindings={'fruit_skin': binding}, asset_sha256=self.identity)
        self.assertEqual(result[7]['colour'], (1., 0., 0.))
        self.assertEqual(result[11]['colour'], (.5, .5, .5))
        self.assertEqual(receipts[0]['texture_sha256'], 'b'*64)

    def test_unverified_foreign_incomplete_texture_rejected(self):
        texture = TextureImage([[[1., 0., 0.]]], 'b'*64)
        binding = TextureBinding('fruit_skin', self.identity, {7: (0., 0.), 11: (0., 0.)}, True)
        for wrong in (replace(binding, binding_verified=False), replace(binding, asset_sha256='c'*64),
                      replace(binding, gaussian_uv={7: (0., 0.)}), replace(binding, gaussian_uv={7: (0., 0.), 99: (0., 0.)})):
            with self.assertRaises(ValueError):
                self.run_ops(TexturePaint('material_worker', 'fruit_skin', 3, 'skin'), textures={'skin': texture},
                             texture_bindings={'fruit_skin': wrong}, asset_sha256=self.identity)

    def test_atomic_rollback_when_later_operation_lacks_authority(self):
        for invalid in (Recolour('intruder', 'fruit_skin', 3), Recolour('material_worker', 'actor', 3),
                        Recolour('material_worker', 'fruit_skin', 2)):
            with self.assertRaises(ValueError):
                self.run_ops(Recolour('material_worker', 'fruit_skin', 3), invalid)
            self.assertEqual(self.state[7]['colour'], [.2, .4, .6])

    def test_field_permission_cannot_be_bypassed_by_typed_operation(self):
        limited = OwnershipManifest(3, (replace(self.part, allowed_fields=frozenset({'colour'})),))
        with self.assertRaises(ValueError):
            apply_agent_operations(self.state, limited, (RigidTransform('material_worker', 'fruit_skin', 3),))

    def test_second_order_translation_and_rotation_reversal_from_canonical(self):
        first = rigid_motion_at('material_worker', 'fruit_skin', 3, time_seconds=1., velocity=(2., 0., 0.),
                                acceleration=(-2., 0., 0.), angular_velocity=(0., 0., math.pi),
                                angular_acceleration=(0., 0., -math.pi), pivot=(7., 0., 0.))
        second = rigid_motion_at('material_worker', 'fruit_skin', 3, time_seconds=2., velocity=(2., 0., 0.),
                                 acceleration=(-2., 0., 0.), angular_velocity=(0., 0., math.pi),
                                 angular_acceleration=(0., 0., -math.pi), pivot=(7., 0., 0.))
        one, _, _ = self.run_ops(first)
        two, _, _ = self.run_ops(second)
        self.assertAlmostEqual(one[11]['position'][0], 8.)
        self.assertAlmostEqual(one[11]['position'][1], 4.)
        self.assertEqual(two[11]['position'], tuple(self.state[11]['position']))
        self.assertEqual(two[11]['covariance'], tuple(tuple(row) for row in self.state[11]['covariance']))

    def test_invalid_rotations_numbers_and_motion_axes_fail(self):
        for rotation in (((2., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                         ((-1., 0., 0.), (0., 1., 0.), (0., 0., 1.))):
            with self.assertRaises(ValueError):
                RigidTransform('material_worker', 'fruit_skin', 3, rotation)
        for value in (True, float('nan'), -1., 61.):
            with self.assertRaises(ValueError):
                rigid_motion_at('material_worker', 'fruit_skin', 3, time_seconds=value)
        with self.assertRaises(ValueError):
            rigid_motion_at('material_worker', 'fruit_skin', 3, time_seconds=1., angular_velocity=(1., 0., 0.), angular_acceleration=(0., 1., 0.))

    def test_strict_json_boundary_and_no_code_execution(self):
        args = dict(kind='recolour', agent_id='material_worker', part_id='fruit_skin', expected_revision=3, rgb=[1., 0., 0.])
        self.assertIsInstance(operation_from_dict(args), Recolour)
        for extra in ({'code': 'print(1)'}, {'gaussian_ids': [99]}, {'path': 'secret'}):
            with self.assertRaises(ValueError):
                operation_from_dict({**args, **extra})
        with self.assertRaises(ValueError):
            operation_from_dict({**args, 'kind': 'execute'})
        with self.assertRaises(ValueError):
            self.run_ops(args)
        with self.assertRaises(ValueError):
            TexturePaint('material_worker', 'fruit_skin', 3, '../texture.png')
        with self.assertRaises(ValueError):
            RigidTransform('material_worker', 'fruit_skin', 3, translation=(1e10, 0., 0.))

    def test_asset_adapter_preserves_ids_and_derives_covariance(self):
        asset = dict(ids=[11, 7], position=[[11., 0., 0.], [7., 0., 0.]], colour=[[.2, .4, .6]]*2,
                     opacity=[.8]*2, frame=[[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]*2, scale=[[1., 2., 3.]]*2)
        state = state_from_asset(asset)
        self.assertEqual(set(state), {7, 11})
        self.assertEqual(state[7]['covariance'], ((1., 0., 0.), (0., 4., 0.), (0., 0., 9.)))
        self.assertNotIn('frame', state[7])
        asset['position'][0][0] = 100.
        self.assertEqual(state[11]['position'][0], 11.)
        with self.assertRaises(ValueError):
            state_from_asset({**asset, 'ids': [7, 7]})

    def test_binding_and_input_aliases_cannot_change_after_validation(self):
        rgb = [1., 0., 0.]
        op = Recolour('material_worker', 'fruit_skin', 3, rgb)
        rgb[0] = 0.
        result, _, _ = self.run_ops(op)
        self.assertEqual(result[7]['colour'], (1., 0., 0.))
        with self.assertRaises(TypeError):
            result[7]['colour'] = (0., 0., 0.)
        with self.assertRaises(ValueError):
            apply_agent_operations(self.state, OwnershipManifest(3, (replace(self.part, binding_verified=False),)), (op,))

    def test_array_export_replays_by_id_not_row(self):
        arrays = state_to_arrays(self.state)
        self.assertEqual(arrays['ids'].tolist(), [7, 11, 99])
        restored = state_from_asset(arrays)
        self.assertEqual(restored[11]['position'], tuple(self.state[11]['position']))

    def test_canonical_npz_binding_is_whole_asset_and_hash_checked(self):
        import hashlib
        from pathlib import Path
        import tempfile
        import numpy as np
        # Two points only; this is a file-contract diagnostic, not local fitting.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'canonical.npz'
            np.savez_compressed(path, position=np.asarray([[0., 0., 0.], [1., 1., 1.]], dtype=np.float32),
                                colour=np.asarray([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            state, manifest, provenance = canonical_state_from_npz(path, expected_sha256=sha,
                                                                  part_id='whole_object', agent_id='object_worker')
            self.assertEqual(set(state), {0, 1})
            self.assertEqual(manifest.parts[0].gaussian_ids, (0, 1))
            self.assertEqual(manifest.parts[0].agent_id, 'object_worker')
            self.assertAlmostEqual(state[0]['covariance'][0][0], .0001)
            self.assertEqual(state[0]['opacity'], .9)
            self.assertFalse(provenance['temporal_scene_correspondence_verified'])
            self.assertFalse(provenance['texture_uv_verified'])
            with self.assertRaises(ValueError):
                canonical_state_from_npz(path, expected_sha256='f'*64, part_id='whole_object', agent_id='object_worker')


if __name__ == '__main__':
    unittest.main()
