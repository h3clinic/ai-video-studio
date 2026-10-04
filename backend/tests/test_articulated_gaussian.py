import unittest

import torch

from real_video.articulated_gaussian import BatchedArticulatedGaussian
from real_video.fit_motion3d import rotation_vectors
from real_video.surface_skin3d import SurfaceSkinner


def example():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [1., 1., 0.], [2., 0., 0.], [2., 1., 0.]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]])
    barycentric = torch.tensor([[.2, .3, .5]]).expand(4, 3).clone()
    asset = dict(mesh_vertices=vertices, mesh_faces=faces, face_id=torch.arange(4),
                 barycentric=barycentric, frame=torch.eye(3).expand(4, 3, 3).clone(),
                 scale=torch.tensor([[.1, .2, .03]]).expand(4, 3).clone())
    rig = dict(joints=torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
               parents=[-1, 0, 1], vertex_weights=torch.tensor([
                   [1., 0., 0.], [.7, .3, 0.], [1., 0., 0.],
                   [.6, .4, 0.], [0., 1., 0.], [0., .9, .1]]))
    return asset, rig


class ArticulatedGaussianTests(unittest.TestCase):
    def test_identity_and_buffers(self):
        a, r = example()
        model = BatchedArticulatedGaussian(a, r)
        result = model(torch.eye(3).expand(2, 3, 3, 3), return_strain=True)
        torch.testing.assert_close(result['vertices'], a['mesh_vertices'][None].expand(2, -1, -1))
        expected = (a['mesh_vertices'][a['mesh_faces']] * a['barycentric'][..., None]).sum(1)
        torch.testing.assert_close(result['position'], expected[None].expand(2, -1, -1))
        torch.testing.assert_close(result['covariance'], model.rest_covariance[None].expand(2, -1, -1, -1))
        torch.testing.assert_close(result['area_ratio'], torch.ones(2, 4))
        torch.testing.assert_close(model.weights(), r['vertex_weights'])
        self.assertEqual(list(dict(model.named_parameters())), ['skinning_logits'])
        self.assertIn('barycentric', dict(model.named_buffers()))

    def test_rigid_covariance_and_translation(self):
        a, r = example()
        model = BatchedArticulatedGaussian(a, r)
        rv = torch.zeros(1, 3, 3)
        rv[0, 0] = torch.tensor([.2, -.4, .6])
        local = rotation_vectors(rv)
        shift = torch.tensor([[.1, -.2, .3]])
        result = model(local, shift)
        q = local[0, 0]
        torch.testing.assert_close(result['vertices'][0], a['mesh_vertices'] @ q.T + shift)
        torch.testing.assert_close(result['covariance'][0], q @ model.rest_covariance @ q.T)

    def test_nonroot_bending_preserves_bones(self):
        a, r = example()
        model = BatchedArticulatedGaussian(a, r)
        rv = torch.zeros(1, 3, 3)
        rv[0, 1, 2] = .7
        result = model(rotation_vectors(rv))
        torch.testing.assert_close(result['joints'][0, 1], r['joints'][1])
        self.assertGreater(float(result['joints'][0, 2, 1]), .6)
        lengths = (result['joints'][:, 1:] - result['joints'][:, :-1]).norm(dim=-1)
        torch.testing.assert_close(lengths, torch.ones(1, 2))
        self.assertGreater(float((result['vertices'][0] - a['mesh_vertices']).abs().max().detach()), .2)

    def test_gradients_reach_pose_covariance_and_skinning(self):
        a, r = example()
        model = BatchedArticulatedGaussian(a, r)
        rv = torch.tensor([[[.1, .2, -.1], [.3, .1, .5], [.1, -.1, .2]]], requires_grad=True)
        shift = torch.tensor([[.1, .2, .3]], requires_grad=True)
        output = model(rotation_vectors(rv), shift)
        loss = output['position'].square().sum() + 30 * output['covariance'].square().sum()
        loss.backward()
        for gradient in [rv.grad, shift.grad, model.skinning_logits.grad]:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 1e-5)
        self.assertEqual(float(model.skinning_logits.grad[~model.support_mask].abs().sum()), 0.)
        # Covariance transport alone must reach pose and skinning, not only centers.
        model.zero_grad()
        rv.grad = None
        model(rotation_vectors(rv))['covariance'].square().sum().backward()
        self.assertGreater(float(rv.grad.abs().sum()), 1e-7)
        self.assertGreater(float(model.skinning_logits.grad.abs().sum()), 1e-7)

    def test_joint_override_gradients_identity_and_bone_lengths(self):
        a, r = example()
        model = BatchedArticulatedGaussian(a, r)
        joints = (r['joints'] + torch.tensor([[0., .1, 0.], [0., .2, .1], [0., -.1, .3]])).requires_grad_()
        identity = torch.eye(3).expand(2, 3, 3, 3)
        torch.testing.assert_close(model(identity, joints=joints)['vertices'], a['mesh_vertices'][None].expand(2, -1, -1))
        rv = torch.tensor([[[0., 0., .2], [0., .3, .5], [.1, 0., .1]]]).expand(2, -1, -1)
        output = model(rotation_vectors(rv), joints=joints)
        observed = (output['joints'][:, 1:] - output['joints'][:, :-1]).norm(dim=-1)
        expected = (joints[1:] - joints[:-1]).norm(dim=-1)
        torch.testing.assert_close(observed, expected[None].expand_as(observed))
        output['position'].square().sum().backward()
        self.assertTrue(torch.isfinite(joints.grad).all())
        self.assertGreater(float(joints.grad.abs().sum()), 1e-5)
        batch_joints = joints.detach()[None].expand(2, -1, -1).clone()
        torch.testing.assert_close(model(rotation_vectors(rv), joints=batch_joints)['vertices'], output['vertices'])

    def test_support_cannot_add_unrelated_joint(self):
        a, r = example()
        model = BatchedArticulatedGaussian(a, r, max_influences=2)
        initial_support = model.support_mask.clone()
        with torch.no_grad():
            model.skinning_logits.add_(torch.randn_like(model.skinning_logits) * 50)
            model.skinning_logits[~initial_support] = 1e9
        weights = model.weights()
        self.assertTrue(torch.equal(weights[~initial_support], torch.zeros_like(weights[~initial_support])))
        self.assertTrue((weights >= 0).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones(len(weights)))
        self.assertEqual(float(weights[:5, 2].sum().detach()), 0.)
        self.assertFalse(BatchedArticulatedGaussian(a, r, train_skinning=False).skinning_logits.requires_grad)

    def test_rendered_only_gradient_reaches_real_network_head_and_skin(self):
        # Procedural unit fixture, not evidence of video quality/generalization.
        from real_video.fit_articulated_weights import ClipJointWeights
        from real_video.gaussian3d import render
        torch.manual_seed(8401)
        a, r = example()
        decoder = BatchedArticulatedGaussian(a, r)
        network = ClipJointWeights(joint_count=3, hidden=16)
        # Off-rest weights are needed: at identity pose all owners agree and
        # skinning has zero effect, hence exactly zero skinning gradient.
        with torch.no_grad():
            network.net[-1].weight.normal_(0, .08)
        local, translation, _ = network(torch.tensor([.65]))
        output = decoder(local, translation)
        colour = torch.tensor([[.9, .2, .1], [.2, .8, .3], [.1, .3, .9], [.8, .7, .2]])
        rgb, alpha = render(output['position'][0], output['covariance'][0],
                            colour, torch.full((4,), .8), torch.tensor([1., .5, 3.]),
                            torch.tensor([1., .5, 0.]), height=48, width=64,
                            radius=4, ground=False)
        x = torch.linspace(0, 1, 64)[None]
        y = torch.linspace(0, 1, 48)[:, None]
        # No pose, joint, covariance, mesh, or parameter loss is added here.
        loss = (rgb[..., 0] * x + rgb[..., 1] * y + alpha * (x + .3*y)).mean()
        loss.backward()
        for name, gradient in [('network_head', network.net[-1].weight.grad),
                               ('network_trunk', network.net[0].weight.grad),
                               ('skinning', decoder.skinning_logits.grad)]:
            with self.subTest(name=name):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 1e-8)

    def test_joint_rotation_does_not_move_unrelated_branch(self):
        a, r = example()
        r['joints'] = torch.cat((r['joints'], torch.tensor([[0., 1., 0.]])))
        r['parents'] = [-1, 0, 1, 0]
        r['vertex_weights'] = torch.cat((r['vertex_weights'], torch.zeros(6, 1)), -1)
        decoder = BatchedArticulatedGaussian(a, r)
        rv = torch.zeros(1, 4, 3)
        rv[0, 1, 2] = .5
        output = decoder(rotation_vectors(rv))
        torch.testing.assert_close(output['joints'][0, [0, 1, 3]], r['joints'][[0, 1, 3]])
        torch.testing.assert_close(output['vertices'][0, [0, 2]], a['mesh_vertices'][[0, 2]])
        self.assertGreater(float((output['joints'][0, 2]-r['joints'][2]).norm()), .4)

    def test_batch_and_subset_equal_single_public_decoder(self):
        torch.manual_seed(15)
        a, r = example()
        model = BatchedArticulatedGaussian(a, r)
        rotation = rotation_vectors(torch.randn(3, 3, 3) * .15)
        translation = torch.randn(3, 3) * .1
        result = model(rotation, translation)
        reference = SurfaceSkinner(a, r)
        for b in range(3):
            old_p, old_c, _, old_v, _ = reference(rotation[b], translation[b])
            torch.testing.assert_close(result['position'][b], old_p)
            torch.testing.assert_close(result['covariance'][b], old_c)
            torch.testing.assert_close(result['vertices'][b], old_v)
            single = model(rotation[b:b+1], translation[b:b+1])
            for key in result:
                torch.testing.assert_close(single[key][0], result[key][b])
        selection = torch.tensor([3, 1, 3])
        subset = model(rotation, translation, gaussian_index=selection)
        torch.testing.assert_close(subset['position'], result['position'][:, selection])
        torch.testing.assert_close(subset['covariance'], result['covariance'][:, selection])

    def test_unused_degenerate_rest_face_safe_owned_rejected(self):
        a, r = example()
        a['mesh_faces'] = torch.cat((a['mesh_faces'], torch.tensor([[0, 0, 0]])))
        model = BatchedArticulatedGaussian(a, r)
        self.assertEqual(model.unused_degenerate_faces, 1)
        output = model(torch.eye(3).expand(1, 3, 3, 3), return_strain=True)
        self.assertTrue(torch.isfinite(output['covariance']).all())
        self.assertEqual(output['area_ratio'].shape, (1, 4))
        a['face_id'][0] = 4
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            BatchedArticulatedGaussian(a, r)

    def test_invalid_values_shapes_indices_rejected(self):
        for key, corrupt in [
                ('barycentric', lambda a: a['barycentric'].fill_(float('nan'))),
                ('mesh_faces', lambda a: a['mesh_faces'].fill_(-1)),
                ('scale', lambda a: a['scale'].zero_())]:
            a, r = example()
            corrupt(a)
            with self.subTest(key=key), self.assertRaises(ValueError):
                BatchedArticulatedGaussian(a, r)
        a, r = example()
        model = BatchedArticulatedGaussian(a, r)
        identity = torch.eye(3).expand(1, 3, 3, 3)
        for kwargs in [dict(gaussian_index=torch.tensor([4])),
                       dict(gaussian_index=torch.tensor([.5])),
                       dict(joints=torch.zeros(1, 2, 3)),
                       dict(translation=torch.full((1, 3), float('nan')))]:
            with self.assertRaises(ValueError):
                model(identity, **kwargs)
        with self.assertRaises(ValueError):
            model(identity * 2)


if __name__ == '__main__':
    unittest.main()
