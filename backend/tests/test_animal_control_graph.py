"""Control-graph numerical contracts, not evidence of bones or animal gait."""
import unittest

import torch

from real_video.animal_control_graph import AnimalControlGraph, bind_points, deform_points, rotate


class AnimalControlGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.reference = torch.tensor([[.1, .2], [.8, .25], [.15, .9], [.9, .8]])
        self.points = torch.tensor([
            [.2, .3, -3., -4., 1., 0., .2, -.1, .5, .8],
            [.6, .4, -4., -3., 0., 1., -.3, .4, .2, 1.],
            [.3, .7, -4., -4., -.6, .8, .1, -.2, -.4, .6],
        ])
        self.index, self.weight = bind_points(self.points[:, :2], self.reference)

    def observations(self):
        pos = self.reference[None, None].repeat(2, 3, 1, 1)
        pos[:, 1] += .005; pos[:, 2] += .015
        angle = torch.tensor([0., .02, .05])[None, :, None].expand(2, 3, 4).clone()
        visible = torch.full((2, 3, 4), .8)
        colour = torch.rand(2, 4, 3) * 2 - 1
        adjacency = torch.rand(2, 4, 4)
        adjacency /= adjacency.sum(-1, keepdim=True)
        return pos, angle, visible, colour, adjacency

    def test_zero_deformation_and_owned_output(self):
        output = deform_points(self.points, self.reference, self.reference,
                               torch.zeros(4), torch.ones(4), self.index, self.weight)
        self.assertTrue(torch.allclose(output, self.points, atol=1e-7))
        output.fill_(100)
        self.assertLess(self.points.abs().max().item(), 100)

    def test_shared_rigid_transform_matches_direct_transform(self):
        angle = torch.tensor(.47)
        translation = torch.tensor([.3, -.2])
        controls = rotate(self.reference, angle) + translation
        result = deform_points(self.points, self.reference, controls, angle.expand(4),
                               torch.ones(4), self.index, self.weight)
        self.assertTrue(torch.allclose(result[:, :2], rotate(self.points[:, :2], angle) + translation, atol=1e-6))
        self.assertTrue(torch.allclose(result[:, 4:6], rotate(self.points[:, 4:6], angle), atol=1e-6))
        self.assertTrue(torch.equal(result[:, 2:4], self.points[:, 2:4]))

    def test_colour_preserved_under_nonrigid_control_changes(self):
        output = deform_points(self.points, self.reference, self.reference + torch.randn(4, 2) * .1,
                               torch.tensor([.2, -.3, .1, .5]), torch.rand(4), self.index, self.weight)
        self.assertTrue(torch.equal(output[:, 6:9], self.points[:, 6:9]))
        self.assertTrue(torch.allclose(output[:, 4:6].norm(dim=-1), torch.ones(3), atol=1e-6))

    def test_visibility_scales_saved_opacity(self):
        for visibility in [torch.zeros(4), torch.ones(4), torch.tensor([.2, .4, .6, .8])]:
            output = deform_points(self.points, self.reference, self.reference, torch.zeros(4),
                                   visibility, self.index, self.weight)
            expected = self.points[:, 9] * (self.weight * visibility[self.index]).sum(1)
            self.assertTrue(torch.allclose(output[:, 9], expected, atol=1e-7))
            self.assertTrue(((output[:, 9] >= 0) & (output[:, 9] <= self.points[:, 9] + 1e-7)).all())

    def test_binding_weights_and_shapes(self):
        for k in [1, 2, 10]:
            index, weight = bind_points(self.points[:, :2], self.reference, k=k)
            self.assertEqual(index.shape, (3, min(k, 4)))
            self.assertEqual(weight.shape, index.shape)
            self.assertTrue(torch.allclose(weight.sum(-1), torch.ones(3)))
            self.assertTrue((weight >= 0).all())

    def test_deformation_has_finite_gradients(self):
        controls = self.reference.clone().requires_grad_()
        angles = torch.tensor([.2, -.3, .1, .5], requires_grad=True)
        visibility = torch.full((4,), .6, requires_grad=True)
        result = deform_points(self.points, self.reference, controls, angles, visibility, self.index, self.weight)
        result.square().sum().backward()
        for value in [controls, angles, visibility]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_learned_step_is_differentiable(self):
        model = AnimalControlGraph(hidden=8)
        state = model.initialize(*self.observations())
        for _ in range(3): state = model.step(state)
        loss = state['position'].square().mean() + state['angle'].square().mean() + state['visibility'].square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.head[-1].weight.grad.abs().sum().item(), 0)

    def test_graph_permutation_equivariance(self):
        model = AnimalControlGraph(hidden=8).eval()
        # Nonzero output weights exercise messages instead of only initialized damping.
        with torch.no_grad(): model.head[-1].weight.normal_(std=.03)
        pos, angle, visible, colour, adjacency = self.observations()
        order = torch.tensor([2, 0, 3, 1])
        state = model.initialize(pos, angle, visible, colour, adjacency)
        permuted = model.initialize(pos[:, :, order], angle[:, :, order], visible[:, :, order],
                                    colour[:, order], adjacency[:, order][:, :, order])
        with torch.no_grad():
            for _ in range(4):
                state = model.step(state); permuted = model.step(permuted)
        for key in ['position', 'reference', 'reference_angle', 'angle', 'velocity', 'acceleration',
                    'angular_velocity', 'visibility', 'colour', 'memory']:
            self.assertTrue(torch.allclose(state[key][:, order], permuted[key], atol=1e-6), key)

    def test_initialize_requires_exactly_three_position_observations(self):
        observed = self.observations()
        model = AnimalControlGraph(hidden=8)
        for length in [1, 2, 4, 15]:
            bad = observed[0][:, :1].repeat(1, length, 1, 1)
            with self.subTest(length=length), self.assertRaises(ValueError):
                model.initialize(bad, *observed[1:])
        state = model.initialize(*observed)
        self.assertTrue(torch.equal(state['position'], observed[0][:, 2]))
        self.assertTrue(torch.equal(state['velocity'], observed[0][:, 2] - observed[0][:, 1]))
        self.assertTrue(torch.equal(state['reference_angle'], observed[1][:, 2]))

    def test_fixed_shape_finite_rollout_and_resume(self):
        model = AnimalControlGraph(hidden=8).eval()
        state = model.initialize(*self.observations())
        shape = {key: value.shape for key, value in state.items()}
        original_colour = state['colour'].clone()
        with torch.no_grad():
            for _ in range(5): state = model.step(state)
            snapshot = {key: value.detach().cpu().clone() for key, value in state.items()}
            for _ in range(8): state = model.step(state)
            resumed = {key: value.clone() for key, value in snapshot.items()}
            for _ in range(8): resumed = model.step(resumed)
            for key in state: self.assertTrue(torch.equal(state[key], resumed[key]), key)
            for _ in range(32): state = model.step(state)
        for key, value in state.items():
            self.assertEqual(value.shape, shape[key], key)
            self.assertTrue(torch.isfinite(value).all(), key)
        self.assertTrue(torch.equal(state['colour'], original_colour))
        self.assertTrue(((state['visibility'] > 0) & (state['visibility'] < 1)).all())

    def test_initialize_rejects_extra_angle_visibility_and_mismatched_shapes(self):
        model = AnimalControlGraph(hidden=8)
        observations = self.observations()
        for field in [1, 2]:
            for length in [1, 2, 4, 15]:
                bad = list(observations)
                bad[field] = bad[field][:, :1].repeat(1, length, 1)
                with self.subTest(field=field, length=length), self.assertRaises(ValueError):
                    model.initialize(*bad)
        for field in [1, 2, 3, 4]:
            bad = list(observations); bad[field] = bad[field][..., :-1]
            with self.subTest(field=field), self.assertRaises(ValueError):
                model.initialize(*bad)

    def test_initializer_owns_colour_and_adjacency(self):
        model = AnimalControlGraph(hidden=8).eval()
        observations = self.observations()
        with torch.no_grad():
            state = model.initialize(*observations)
            expected = model.step(state)
            colour = state['colour'].clone(); adjacency = state['adjacency'].clone()
            observations[3].fill_(100); observations[4].fill_(0)
            actual = model.step(state)
        self.assertTrue(torch.equal(state['colour'], colour))
        self.assertTrue(torch.equal(state['adjacency'], adjacency))
        for key in expected: self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_binding_rejects_nonpositive_k_empty_reference_and_wrong_shapes(self):
        for k in [0, -1, -4]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                bind_points(self.points[:, :2], self.reference, k=k)
        with self.assertRaises(ValueError): bind_points(self.points[:, :2], self.reference[:0])
        with self.assertRaises(ValueError): bind_points(self.points[:, :3], self.reference)
        with self.assertRaises(ValueError): bind_points(self.points[:, :2], self.reference[:, :1])

    def test_mask_rejects_cross_gap_binding_and_fallback_is_normalized(self):
        # Nonsquare image tests image-height coordinates, not width-normalized x.
        mask = torch.ones(32, 64); mask[:, 20:40] = 0
        point = torch.tensor([[.3, .5]])
        controls = torch.tensor([[.5, .5], [1.5, .5]])
        index, weight = bind_points(point, controls, k=2, mask=mask)
        self.assertEqual(weight[index == 1].item(), 0)
        self.assertEqual(weight[index == 0].item(), 1)
        index, weight = bind_points(point, controls, k=2, mask=torch.zeros_like(mask))
        self.assertTrue(torch.equal(weight, torch.tensor([[1., 0.]])))
        self.assertEqual(index[0, 0].item(), 0)
        self.assertTrue(torch.equal(weight.sum(-1), torch.ones(1)))
        self.assertTrue(torch.isfinite(weight).all())

    def test_opposing_rotations_have_finite_gradients_and_unit_axes(self):
        points = self.points[:1].clone()
        reference = self.reference[:2]
        angle = torch.tensor([0., torch.pi], requires_grad=True)
        weights = torch.tensor([[.5, .5]], requires_grad=True)
        output = deform_points(points, reference, reference, angle, torch.ones(2),
                               torch.tensor([[0, 1]]), weights)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output[:, 4:6].norm(dim=-1), torch.ones(1), atol=1e-6))
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(angle.grad).all())
        self.assertTrue(torch.isfinite(weights.grad).all())

    def test_deformation_rejects_invalid_visibility_and_weights(self):
        for invalid in [-.01, 1.01, float('nan'), float('inf')]:
            visible = torch.ones(4); visible[0] = invalid
            with self.subTest(visibility=invalid), self.assertRaises(ValueError):
                deform_points(self.points, self.reference, self.reference, torch.zeros(4),
                              visible, self.index, self.weight)
        cases = [torch.zeros_like(self.weight), self.weight * 2]
        for invalid in [-.1, float('nan'), float('inf')]:
            weight = self.weight.clone(); weight[0, 0] = invalid; cases.append(weight)
        for weight in cases:
            with self.subTest(weight=weight.tolist()), self.assertRaises(ValueError):
                deform_points(self.points, self.reference, self.reference, torch.zeros(4),
                              torch.ones(4), self.index, weight)


if __name__ == '__main__': unittest.main()
