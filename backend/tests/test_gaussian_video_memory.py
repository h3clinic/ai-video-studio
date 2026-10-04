"""Small CPU contract tests, not visual motion/anatomy quality claims."""
import tempfile
import unittest
from pathlib import Path

import torch

from real_video.gaussian_video_memory import GaussianVideoMemory


def asset():
    return dict(position=torch.tensor([[-.12, 0., 0.], [.12, 0., 0.]]),
                covariance=torch.diag(torch.tensor([.002, .001, .0005])).repeat(2, 1, 1),
                colour=torch.tensor([[.9, .3, .1], [.8, .5, .2]]),
                opacity=torch.tensor([.8, .9]), ids=torch.tensor([71, 93]),
                normal=torch.tensor([[0., 0., 1.], [0., 0., 1.]]))


class GaussianVideoMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_canonical_transport_not_cumulative_and_fixed_appearance(self):
        original = asset()
        memory = GaussianVideoMemory(original)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).repeat(2, 1, 1)
        p = original['position'] @ rotation[0].T
        for _ in range(2):
            memory.update(p, ids=original['ids'], dt=.1, rotations=rotation)
        snapshot = memory.snapshot()
        torch.testing.assert_close(snapshot['covariance'], rotation @ original['covariance'] @ rotation.transpose(-1, -2))
        torch.testing.assert_close(snapshot['position'], p)
        digest = memory.asset_digest
        original['colour'].zero_()
        original['ids'][0] = 999
        self.assertEqual(memory.asset_digest, digest)
        self.assertFalse(torch.equal(memory._asset['colour'], original['colour']))
        self.assertEqual(memory._asset['ids'].tolist(), [71, 93])

    def test_bounded_reused_buffers_and_no_future_history(self):
        original = asset()
        memory = GaussianVideoMemory(original)
        identity = torch.eye(3).repeat(2, 1, 1)
        before = memory.memory_bytes()
        pointers = {memory._position.data_ptr(), memory._previous_position.data_ptr()}
        canonical = {key: value.data_ptr() for key, value in memory._asset.items()}
        for t in range(100):
            memory.update(original['position'] + t*.001, ids=original['ids'], dt=1/30, rotations=identity)
            self.assertEqual({memory._position.data_ptr(), memory._previous_position.data_ptr()}, pointers)
            self.assertEqual({key: value.data_ptr() for key, value in memory._asset.items()}, canonical)
        self.assertEqual(memory.memory_bytes(), before)
        self.assertEqual(before['frame_history_tensor_bytes'], 0)
        self.assertEqual(before['stored_future_trajectory_tensor_bytes'], 0)
        tensors = [v for v in memory.snapshot().values() if isinstance(v, torch.Tensor)]
        self.assertEqual(sum(v.numel()*v.element_size() for v in tensors), before['snapshot_tensor_bytes'])

    def test_exact_serialized_resume_and_snapshot_ownership(self):
        original = asset()
        a, b = GaussianVideoMemory(original), GaussianVideoMemory(original)
        gradient = torch.diag(torch.tensor([1.2, .9, 1.])).repeat(2, 1, 1)
        a.update(original['position']+.1, ids=original['ids'], dt=.2, deformation_gradient=gradient)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.pt'
            a.save(path)
            b.load(path)
        for memory in (a, b):
            memory.update(original['position']+.2, ids=original['ids'], dt=.1, deformation_gradient=gradient)
        for key, value in a.snapshot().items():
            other = b.snapshot()[key]
            self.assertTrue(torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)
        snapshot = a.snapshot()
        snapshot['position'].zero_()
        self.assertFalse(torch.equal(a.snapshot()['position'], snapshot['position']))

    def test_shear_transports_covariance_and_material_normal(self):
        original = asset()
        original['normal'] = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        memory = GaussianVideoMemory(original)
        gradient = torch.tensor([[1., .3, 0.], [0., 1., 0.], [0., 0., 1.]]).repeat(2, 1, 1)
        position = (gradient @ original['position'][..., None])[..., 0]
        memory.update(position, ids=original['ids'], dt=.1, deformation_gradient=gradient)
        snapshot = memory.snapshot()
        expected_normal = torch.tensor([[1., -.3, 0.], [1., -.3, 0.]])
        expected_normal /= expected_normal.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(snapshot['normal'], expected_normal)
        torch.testing.assert_close(snapshot['covariance'], gradient @ original['covariance'] @ gradient.transpose(-1, -2))

    def test_missing_normal_snapshot_can_resume(self):
        original = asset()
        del original['normal']
        a, b = GaussianVideoMemory(original), GaussianVideoMemory(original)
        self.assertFalse(a.normals_valid)
        b.restore(a.snapshot())
        self.assertFalse(b.normals_valid)

    def test_invalid_updates_fail_before_state_mutation(self):
        original = asset()
        memory = GaussianVideoMemory(original)
        identity = torch.eye(3).repeat(2, 1, 1)
        cases = [dict(ids=original['ids'].flip(0), dt=.1, rotations=identity),
                 dict(ids=original['ids'], dt=0, rotations=identity),
                 dict(ids=original['ids'], dt=.1, rotations=identity*2),
                 dict(ids=original['ids'], dt=.1, deformation_gradient=identity*0),
                 dict(ids=original['ids'], dt=.1, deformation_gradient=-identity),
                 dict(ids=original['ids'], dt=.1, rotations=identity*float('nan')),
                 dict(ids=original['ids'], dt=.1, rotations=identity, deformation_gradient=identity)]
        before = memory.snapshot()
        for arguments in cases:
            with self.assertRaises(ValueError):
                memory.update(original['position'], **arguments)
            self.assertEqual(memory.step_index, 0)
            self.assertTrue(torch.equal(memory.snapshot()['position'], before['position']))
        with self.assertRaises(ValueError):
            memory.update(original['position'][:, :2], ids=original['ids'], dt=.1, rotations=identity)

    def test_covariance_output_requires_normal_provenance(self):
        original = asset()
        memory = GaussianVideoMemory(original)
        memory.update_covariance(original['position'], original['covariance'], ids=original['ids'], dt=.1)
        self.assertFalse(memory.normals_valid)
        self.assertEqual(memory.snapshot()['normal'].count_nonzero(), 0)
        with self.assertRaises(ValueError):
            memory.update_covariance(original['position'], original['covariance']*0, ids=original['ids'], dt=.1)
        memory.update_covariance(original['position'], original['covariance'], ids=original['ids'], dt=.1,
                                 normal=original['normal'])
        self.assertTrue(memory.normals_valid)

    def test_wrong_asset_duplicate_ids_and_bad_snapshot_rejected(self):
        original = asset()
        memory = GaussianVideoMemory(original)
        changed = asset()
        changed['colour'][0, 0] = .1
        with self.assertRaises(ValueError):
            GaussianVideoMemory(changed).restore(memory.snapshot())
        changed['ids'][1] = changed['ids'][0]
        with self.assertRaises(ValueError):
            GaussianVideoMemory(changed)
        bad = memory.snapshot()
        bad['position'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            memory.restore(bad)

    def test_cpu_projection_current_motion_and_empty_visibility(self):
        original = asset()
        memory = GaussianVideoMemory(original)
        camera = dict(eye=torch.tensor([0., 0., 3.]), target=torch.zeros(3), fov=42.)
        first = memory.project_current(camera, 32, 48, downsample=8, include_velocity=True)
        self.assertEqual(first['features'].shape, (1, 17, 4, 6))
        self.assertTrue(torch.isfinite(first['features']).all())
        self.assertEqual(first['features'][:, 14:].count_nonzero(), 0)
        memory.update(original['position']+.1, ids=original['ids'], dt=.1, rotations=torch.eye(3).repeat(2, 1, 1))
        second = memory.project_current(camera, 32, 48, include_velocity=True)
        self.assertGreater(second['features'][:, 14:].abs().sum(), 0)
        self.assertGreater((second['features']-first['features']).abs().sum(), 0)
        reverse = dict(eye=torch.tensor([0., 0., 3.]), target=torch.tensor([0., 0., 4.]), fov=42.)
        hidden = memory.project_current(reverse, 32, 48)
        self.assertEqual(hidden['alpha'].count_nonzero(), 0)
        self.assertEqual(memory.count, 2)  # Occlusion never deletes identity memory.


if __name__ == '__main__':
    unittest.main()
