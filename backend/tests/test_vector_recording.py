"""Independent arithmetic contracts, not evidence of physically exact motion."""
import unittest

import torch

from real_video.gaussian3d import rotation
from real_video.vector_recording import (
    GaussianSurfaceMemory, record_sequence, recording_audit,
)


def example():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                             [0., 1., 0.], [1., 1., 0.]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]])
    barycentric = torch.tensor([[.2, .3, .5], [0., .5, .5], [.5, 0., .5]])
    face_id = torch.tensor([0, 0, 1])
    frame = rotation(torch.tensor([0., 0., 1.]), torch.tensor(.23))
    return dict(mesh_vertices=vertices, mesh_faces=faces,
                face_id=face_id, barycentric=barycentric,
                ids=torch.tensor([43, 7, 912]),
                position=(vertices[faces[face_id]] * barycentric[..., None]).sum(1),
                frame=frame[None].repeat(3, 1, 1),
                scale=torch.tensor([[.13, .07, .02]]).repeat(3, 1),
                colour=torch.tensor([[.1, .4, .2], [.5, .7, .9], [.2, .8, .6]]),
                opacity=torch.tensor([.7, .8, .9]))


class VectorRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_rest_decode_matches_canonical_mean_covariance_and_frame(self):
        asset = example()
        memory = GaussianSurfaceMemory(asset)
        decoded = memory.decode(asset['mesh_vertices'])
        torch.testing.assert_close(decoded['position'], asset['position'])
        torch.testing.assert_close(decoded['frame'], asset['frame'])
        expected = (asset['frame'] * asset['scale'][:, None].square()) @ asset['frame'].transpose(-1, -2)
        torch.testing.assert_close(decoded['covariance'], expected)
        torch.testing.assert_close(decoded['area_ratio'], torch.ones(2))

    def test_rigid_transport_obeys_world_change_of_basis(self):
        asset = example()
        memory = GaussianSurfaceMemory(asset)
        q = rotation(torch.tensor([.2, .7, -.4]), torch.tensor(.8))
        translation = torch.tensor([.3, -.2, .7])
        decoded = memory.decode(asset['mesh_vertices'] @ q.T + translation)
        torch.testing.assert_close(decoded['position'], asset['position'] @ q.T + translation)
        torch.testing.assert_close(decoded['frame'], q @ asset['frame'], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(decoded['covariance'], q @ memory.rest_covariance @ q.T)

    def test_shared_edge_gaussians_cannot_separate(self):
        asset = example()
        vertices = asset['mesh_vertices'].clone()
        vertices[3] += torch.tensor([.17, -.2, .31])
        decoded = GaussianSurfaceMemory(asset).decode(vertices)
        torch.testing.assert_close(decoded['position'][1], decoded['position'][2])

    def test_shear_keeps_proper_material_frames_and_full_covariance(self):
        asset = example()
        affine = torch.tensor([[1.1, .7, 0.], [0., .8, 0.], [0., 0., 1.]])
        memory = GaussianSurfaceMemory(asset)
        decoded = memory.decode(asset['mesh_vertices'] @ affine.T)
        expected = affine @ memory.rest_covariance @ affine.T
        torch.testing.assert_close(decoded['covariance'], expected)
        frame = decoded['frame']
        torch.testing.assert_close(frame.transpose(-1, -2) @ frame, torch.eye(3).repeat(3, 1, 1), atol=2e-6, rtol=2e-6)
        self.assertTrue((torch.det(frame) > .99999).all())
        self.assertTrue((torch.linalg.eigvalsh(decoded['covariance']) > 0).all())
        # Material orientation is NOT a covariance eigenframe under shear.
        material_covariance = frame.transpose(-1, -2) @ expected @ frame
        self.assertGreater(float(material_covariance[:, 0, 1].abs().max()), 1e-4)

    def test_translation_record_has_explicit_intervals_and_exact_velocity(self):
        asset = example()
        step = torch.tensor([.1, -.04, .03])
        vertices = asset['mesh_vertices'][None] + torch.arange(6)[:, None, None] * step
        record = record_sequence(asset, vertices, fps=16.)
        self.assertTrue(torch.equal(record['ids'], asset['ids']))
        self.assertTrue(torch.equal(record['interval_start'], torch.arange(5)))
        self.assertEqual(record['delta'].shape, (5, 3, 3))
        torch.testing.assert_close(record['delta'], step.expand(5, 3, 3))
        torch.testing.assert_close(record['velocity'], (16 * step).expand(5, 3, 3))
        torch.testing.assert_close(record['angular_velocity'], torch.zeros(5, 3, 3), atol=3e-6, rtol=0)
        audit = recording_audit(record)
        self.assertLess(audit['position_replay_max_error'], 1e-6)
        self.assertLess(audit['rotation_step_max_error'], 1e-6)
        expected_bytes = sum(v.numel() * v.element_size() for v in record.values() if isinstance(v, torch.Tensor))
        self.assertEqual(audit['expanded_tensor_bytes'], expected_bytes)

    def test_world_angular_velocity_uses_left_increment_not_local_increment(self):
        asset = example()
        q0 = rotation(torch.tensor([1., 0., 0.]), torch.tensor(.8))
        left_increment = rotation(torch.tensor([0., 0., 1.]), torch.tensor(.35))
        q1 = left_increment @ q0
        vertices = torch.stack([asset['mesh_vertices'] @ q.T for q in (q0, q1)])
        record = record_sequence(asset, vertices, fps=20.)
        expected = torch.tensor([0., 0., 7.]).expand(1, 3, 3)
        torch.testing.assert_close(record['angular_velocity'], expected, atol=5e-6, rtol=2e-6)
        self.assertLess(recording_audit(record)['rotation_step_max_error'], 1e-6)

    def test_near_pi_rotation_stays_finite_and_replays(self):
        asset = example()
        q = rotation(torch.tensor([.3, -.5, .8]), torch.tensor(torch.pi - 1e-5))
        record = record_sequence(asset, torch.stack([asset['mesh_vertices'], asset['mesh_vertices'] @ q.T]), fps=24.)
        self.assertTrue(torch.isfinite(record['angular_velocity']).all())
        self.assertLess(recording_audit(record)['rotation_step_max_error'], 2e-6)

    def test_id_order_is_not_sorted_and_records_do_not_alias_asset_or_inputs(self):
        asset = example()
        vertices = asset['mesh_vertices'][None].repeat(2, 1, 1)
        copies = {k: v.clone() for k, v in asset.items() if isinstance(v, torch.Tensor)}
        record = record_sequence(asset, vertices, fps=24.)
        self.assertEqual(record['ids'].tolist(), [43, 7, 912])
        record['ids'].fill_(-1)
        record['position'].fill_(999)
        record['frame'].fill_(999)
        record['covariance'].fill_(999)
        for key, expected in copies.items():
            self.assertTrue(torch.equal(asset[key], expected), key)
        self.assertTrue(torch.equal(vertices, asset['mesh_vertices'][None].repeat(2, 1, 1)))

    def test_decode_rejects_wrong_vertex_shape_and_nonfinite_vertices(self):
        asset = example()
        memory = GaussianSurfaceMemory(asset)
        with self.assertRaises(ValueError):
            memory.decode(asset['mesh_vertices'][:-1])
        for invalid in (float('nan'), float('inf')):
            vertices = asset['mesh_vertices'].clone()
            vertices[0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                memory.decode(vertices)

    def test_duplicate_identity_is_rejected(self):
        asset = example()
        asset['ids'][1] = asset['ids'][0]
        with self.assertRaises(ValueError):
            GaussianSurfaceMemory(asset)

    def test_identity_count_must_match_every_gaussian_attribute(self):
        for field in ('ids', 'face_id', 'barycentric', 'frame', 'scale'):
            asset = example()
            asset[field] = asset[field][:-1]
            with self.subTest(field=field), self.assertRaises(ValueError):
                GaussianSurfaceMemory(asset)

    def test_invalid_face_indices_raise_clear_value_error(self):
        for field, invalid in (('mesh_faces', -1), ('mesh_faces', 100),
                               ('face_id', -1), ('face_id', 100)):
            asset = example()
            asset[field].flatten()[0] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                GaussianSurfaceMemory(asset)

    def test_attachment_weights_must_be_nonnegative_finite_partition(self):
        for invalid in (-.1, float('nan'), float('inf')):
            asset = example()
            asset['barycentric'][0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GaussianSurfaceMemory(asset)
        asset = example()
        asset['barycentric'][0] *= 2
        with self.assertRaises(ValueError):
            GaussianSurfaceMemory(asset)

    def test_scales_must_be_strictly_positive_and_finite(self):
        for invalid in (0., -.1, float('nan'), float('inf')):
            asset = example()
            asset['scale'][0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GaussianSurfaceMemory(asset)

    def test_canonical_frames_must_be_proper_orthonormal(self):
        for invalid in ('reflection', 'stretch', 'nan'):
            asset = example()
            if invalid == 'reflection':
                asset['frame'][0, :, 0] *= -1
            elif invalid == 'stretch':
                asset['frame'][0, :, 0] *= 2
            else:
                asset['frame'][0, 0, 0] = float('nan')
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GaussianSurfaceMemory(asset)

    def test_owned_face_collapse_is_rejected(self):
        asset = example()
        memory = GaussianSurfaceMemory(asset)
        vertices = asset['mesh_vertices'].clone()
        vertices[2] = vertices[0]
        with self.assertRaises(ValueError):
            memory.decode(vertices)
        asset['mesh_vertices'] = vertices
        with self.assertRaises(ValueError):
            GaussianSurfaceMemory(asset)

    def test_unused_degenerate_faces_do_not_poison_owned_records(self):
        asset = example()
        asset['mesh_faces'] = torch.cat([asset['mesh_faces'], torch.tensor([[0, 0, 0]])])
        decoded = GaussianSurfaceMemory(asset).decode(asset['mesh_vertices'])
        self.assertTrue(torch.isfinite(decoded['covariance']).all())
        self.assertEqual(len(decoded['area_ratio']), 2)

    def test_record_rejects_invalid_fps_or_insufficient_states(self):
        asset = example()
        vertices = asset['mesh_vertices'][None].repeat(2, 1, 1)
        for fps in (0, -1, float('nan'), float('inf')):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                record_sequence(asset, vertices, fps)
        for invalid in (vertices[:0], vertices[:1], asset['mesh_vertices']):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                record_sequence(asset, invalid, 24.)

    def test_record_interval_integration_reconstructs_nonuniform_trajectory(self):
        asset = example()
        states = []
        for index in range(9):
            q = rotation(torch.tensor([.2, .8, .3]), torch.tensor(.015 * index * index))
            shift = torch.tensor([.002 * index * index, -.01 * index, .003 * index ** 2])
            states.append(asset['mesh_vertices'] @ q.T + shift)
        record = record_sequence(asset, torch.stack(states), fps=16.)
        audit = recording_audit(record)
        self.assertLess(audit['position_replay_max_error'], 2e-7)
        self.assertLess(audit['rotation_step_max_error'], 1e-6)
        self.assertLess(audit['orthogonality_max_error'], 1e-6)
        self.assertGreater(audit['minimum_frame_determinant'], .99999)

    def test_acceleration_times_are_internal_states_not_velocity_midpoints(self):
        asset = example()
        record = record_sequence(asset, asset['mesh_vertices'][None].repeat(6, 1, 1), 8.)
        torch.testing.assert_close(record['state_time_seconds'], torch.arange(6, dtype=torch.float64) / 8)
        torch.testing.assert_close(record['velocity_time_seconds'], (torch.arange(5, dtype=torch.float64) + .5) / 8)
        torch.testing.assert_close(record['acceleration_time_seconds'], torch.arange(1, 5, dtype=torch.float64) / 8)
        self.assertTrue(torch.equal(record['acceleration_sample_index'], torch.arange(1, 5)))
        self.assertEqual(record['acceleration'].shape, (4, 3, 3))
        self.assertEqual(record['angular_acceleration'].shape, (4, 3, 3))

    def test_quadratic_translation_has_acceleration_in_units_per_second_squared(self):
        asset = example()
        curvature = torch.tensor([.003, -.002, .005])
        states = asset['mesh_vertices'][None] + torch.arange(7)[:, None, None].square() * curvature
        slow = record_sequence(asset, states, 10.)
        fast = record_sequence(asset, states, 20.)
        expected = (2 * curvature * 100).expand(5, 3, 3)
        torch.testing.assert_close(slow['acceleration'], expected, atol=3e-5, rtol=0)
        torch.testing.assert_close(fast['acceleration'], slow['acceleration'] * 4, atol=1e-6, rtol=0)
        audit = recording_audit(slow)
        self.assertLess(audit['linear_velocity_replay_max_error'], 1e-6)
        self.assertEqual(audit['linear_acceleration_definition_max_error'], 0.)

    def test_changing_left_rotation_has_world_angular_acceleration(self):
        asset = example()
        q0 = rotation(torch.tensor([1., 0., 0.]), torch.tensor(.8))
        states = []
        for index in range(7):
            left = rotation(torch.tensor([0., 0., 1.]), torch.tensor(.001 * index * index))
            states.append(asset['mesh_vertices'] @ (left @ q0).T)
        record = record_sequence(asset, torch.stack(states), 20.)
        expected = torch.tensor([0., 0., .8]).expand(5, 3, 3)
        torch.testing.assert_close(record['angular_acceleration'], expected, atol=2e-4, rtol=0)
        audit = recording_audit(record)
        self.assertLess(audit['angular_velocity_replay_max_error'], 1e-6)
        self.assertEqual(audit['angular_acceleration_definition_max_error'], 0.)

    def test_two_states_have_valid_empty_accelerations_and_clocks(self):
        asset = example()
        record = record_sequence(asset, asset['mesh_vertices'][None].repeat(2, 1, 1), 16.)
        for key in ('acceleration', 'angular_acceleration'):
            self.assertEqual(record[key].shape, (0, 3, 3))
        self.assertEqual(record['acceleration_time_seconds'].shape, (0,))
        audit = recording_audit(record)
        self.assertTrue(audit['linear_acceleration_available'])
        self.assertEqual(audit['linear_acceleration_samples'], 0)
        self.assertEqual(audit['linear_velocity_replay_max_error'], 0.)
        self.assertEqual(audit['angular_velocity_replay_max_error'], 0.)

    def test_old_record_without_acceleration_or_times_remains_auditable(self):
        asset = example()
        record = record_sequence(asset, asset['mesh_vertices'][None].repeat(4, 1, 1), 16.)
        for key in ('acceleration', 'angular_acceleration', 'acceleration_sample_index',
                    'state_time_seconds', 'velocity_time_seconds', 'acceleration_time_seconds', 'record_version'):
            del record[key]
        audit = recording_audit(record)
        self.assertFalse(audit['linear_acceleration_available'])
        self.assertFalse(audit['angular_acceleration_available'])
        self.assertIsNone(audit['linear_velocity_replay_max_error'])
        self.assertIsNone(audit['angular_velocity_replay_max_error'])

    def test_acceleration_tampering_is_detected_by_replay_and_definition_errors(self):
        asset = example()
        record = record_sequence(asset, asset['mesh_vertices'][None].repeat(5, 1, 1), 16.)
        for key, prefix in (('acceleration', 'linear'), ('angular_acceleration', 'angular')):
            record[key][0, 0, 0] = 2.
            audit = recording_audit(record)
            self.assertGreater(audit[f'{prefix}_velocity_replay_max_error'], .1)
            self.assertEqual(audit[f'{prefix}_acceleration_definition_max_error'], 2.)

    def test_audit_rejects_nonfinite_or_wrong_shaped_acceleration(self):
        asset = example()
        for key in ('acceleration', 'angular_acceleration'):
            for invalid in ('nonfinite', 'shape'):
                record = record_sequence(asset, asset['mesh_vertices'][None].repeat(5, 1, 1), 16.)
                if invalid == 'nonfinite':
                    record[key][0, 0, 0] = float('nan')
                else:
                    record[key] = record[key][:-1]
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    recording_audit(record)

    def test_audit_rejects_wrong_acceleration_time_assignment(self):
        asset = example()
        record = record_sequence(asset, asset['mesh_vertices'][None].repeat(5, 1, 1), 16.)
        record['acceleration_time_seconds'] += .5 / 16
        with self.assertRaises(ValueError):
            recording_audit(record)

    def test_audit_and_record_reject_invalid_scalar_fps(self):
        asset = example()
        states = asset['mesh_vertices'][None].repeat(3, 1, 1)
        record = record_sequence(asset, states, 16.)
        for value in (True, None, [], 'not a number', 0., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                record_sequence(asset, states, value)
            record['fps'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                recording_audit(record)


if __name__ == '__main__':
    unittest.main()
