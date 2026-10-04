"""Numerical and anti-freezing checks, not proof of semantic articulation."""
import unittest

import torch

from real_video.motion_coherence import (
    coherence_loss, edges, geometry_metrics, make_context, passes_gate,
    signed_area, weighted_mean,
)


class MotionCoherenceTests(unittest.TestCase):
    def fields(self, batch=2, height=4, width=5):
        yy, xx = torch.meshgrid((torch.arange(height) + .5) / height,
                                (torch.arange(width) + .5) / width, indexing='ij')
        fields = torch.zeros(batch, 9, height, width)
        fields[:, :2] = torch.stack((xx, yy))
        fields[:, 2:4] = -4
        fields[:, 4] = 1
        return fields

    def context(self, fields):
        return make_context(fields, fields, fields)

    def test_identical_targets_have_zero_loss_and_metrics(self):
        observed = self.fields()
        target = observed.clone()
        target[:, 0] *= 1.3
        target[:, 1] *= .7
        ctx = self.context(observed)
        self.assertEqual(coherence_loss(target, target, ctx).item(), 0)
        for key, value in geometry_metrics(target, target, ctx).items():
            self.assertEqual(value.item(), 0, key)

    def test_edge_supervision_is_invariant_to_independent_translation(self):
        observed = self.fields()
        target = observed.clone(); predicted = observed.clone()
        predicted[:, 0, 1, 2] += .025
        ctx = self.context(observed)
        original = coherence_loss(predicted, target, ctx)
        predicted[:, :2] += torch.tensor([.4, -.2])[None, :, None, None]
        target[:, :2] += torch.tensor([-.3, .6])[None, :, None, None]
        self.assertTrue(torch.allclose(original, coherence_loss(predicted, target, ctx), atol=1e-5))

    def test_fold_penalty_applies_only_where_target_retains_orientation(self):
        target = self.fields()
        predicted = target.clone(); predicted[:, 0] *= -1
        ctx = self.context(target)
        metrics = geometry_metrics(predicted, target, ctx)
        edge_error = metrics['edge_mse_pixels64']
        self.assertEqual(metrics['fold_fraction'].item(), 1)
        self.assertEqual(metrics['target_fold_fraction'].item(), 0)
        self.assertEqual(metrics['fold_mismatch'].item(), 1)
        self.assertTrue(torch.allclose(coherence_loss(predicted, target, ctx) - edge_error,
                                       torch.tensor(.1 * 1.05 ** 2), atol=1e-4))
        self.assertEqual(coherence_loss(predicted, predicted, ctx).item(), 0)
        matched = geometry_metrics(predicted, predicted, ctx)
        self.assertEqual(matched['fold_mismatch'].item(), 0)

    def test_reference_orientation_can_be_negative(self):
        observed = self.fields(); observed[:, 0] *= -1
        ctx = self.context(observed)
        self.assertEqual(coherence_loss(observed, observed, ctx).item(), 0)
        self.assertEqual(geometry_metrics(observed, observed, ctx)['fold_fraction'].item(), 0)

    def test_loss_has_finite_nonzero_position_gradients(self):
        observed = self.fields()
        predicted = observed.clone()
        predicted[:, 0, 1, 2] += .04
        predicted.requires_grad_()
        value = coherence_loss(predicted, observed, self.context(observed))
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(predicted.grad[:, :2].abs().sum().item(), 0)
        self.assertEqual(predicted.grad[:, 2:].abs().sum().item(), 0)

    def test_colour_boundary_downweights_only_boundary_edges(self):
        observed = self.fields()
        observed[:, 6:9, :, 2:] = 1
        wx, wy = self.context(observed)['edge_weights']
        self.assertTrue((wx[..., 1] < 1e-6).all())
        self.assertTrue(torch.equal(wx[..., 0], torch.ones_like(wx[..., 0])))
        self.assertTrue(torch.equal(wx[..., 2:], torch.ones_like(wx[..., 2:])))
        self.assertTrue(torch.equal(wy, torch.ones_like(wy)))

    def test_velocity_boundary_downweights_and_context_is_detached(self):
        first = self.fields(); second = first.clone(); third = first.clone()
        third[:, 0, :, 2:] += .1
        third.requires_grad_()
        ctx = make_context(first, second, third)
        self.assertTrue((ctx['edge_weights'][0][..., 1] < .001).all())
        for value in [*ctx['edge_weights'], ctx['cell_weight'], ctx['reference_area']]:
            self.assertFalse(value.requires_grad)

    def test_zero_weights_have_finite_zero_mean_and_gradients(self):
        value = torch.tensor([1., -2., 3.], requires_grad=True)
        result = weighted_mean(value, torch.zeros_like(value))
        self.assertEqual(result.item(), 0)
        result.backward()
        self.assertTrue(torch.equal(value.grad, torch.zeros_like(value)))
        empty = weighted_mean(torch.empty(0), torch.empty(0))
        self.assertEqual(empty.item(), 0)

    def test_degenerate_observation_cells_are_masked_safely(self):
        observed = self.fields(); observed[:, :2] = 0
        ctx = self.context(observed)
        self.assertEqual(ctx['cell_weight'].sum().item(), 0)
        self.assertTrue(torch.isfinite(ctx['reference_area']).all())
        self.assertEqual(coherence_loss(observed, observed, ctx).item(), 0)
        for value in geometry_metrics(observed, observed, ctx).values():
            self.assertTrue(torch.isfinite(value))

    def baseline(self):
        return dict(edge_mse_pixels64=1., weighted_position_epe_pixels64=2.,
                    rgb_mse=.02, fold_mismatch=.01, mean_displacement_pixels64=3.)

    def test_gate_rejects_freezing_despite_better_other_metrics(self):
        baseline = self.baseline()
        frozen = {key: 0. for key in baseline}
        result = passes_gate(frozen, baseline)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['motion_retained'])
        self.assertTrue(all(value for key, value in result['checks'].items() if key != 'motion_retained'))

    def test_gate_requires_each_predeclared_guard(self):
        baseline = self.baseline()
        candidate = dict(baseline, edge_mse_pixels64=.9)
        self.assertTrue(passes_gate(candidate, baseline)['passed'])
        violations = dict(edge_mse_pixels64=.951, weighted_position_epe_pixels64=2.041,
                          rgb_mse=.02021, fold_mismatch=.01201, mean_displacement_pixels64=2.399)
        for key, value in violations.items():
            with self.subTest(key=key):
                self.assertFalse(passes_gate(dict(candidate, **{key: value}), baseline)['passed'])
                self.assertFalse(passes_gate(dict(candidate, **{key: float('nan')}), baseline)['passed'])

    def test_shapes_and_invalid_context_inputs(self):
        observed = self.fields(batch=2, height=4, width=5)
        ex, ey = edges(observed[:, :2])
        self.assertEqual(ex.shape, (2, 2, 4, 4))
        self.assertEqual(ey.shape, (2, 2, 3, 5))
        self.assertEqual(signed_area(observed[:, :2]).shape, (2, 1, 3, 4))
        ctx = self.context(observed)
        self.assertEqual(ctx['edge_weights'][0].shape, (2, 1, 4, 4))
        self.assertEqual(ctx['edge_weights'][1].shape, (2, 1, 3, 5))
        self.assertEqual(ctx['cell_weight'].shape, (2, 1, 3, 4))
        for malformed in [observed[:, :8], observed[0], observed[..., :3]]:
            with self.subTest(shape=malformed.shape), self.assertRaises(ValueError):
                make_context(observed, observed, malformed)


if __name__ == '__main__': unittest.main()
