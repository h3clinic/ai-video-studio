import unittest
import torch

from real_video.wan_gaussian import WanGaussianDecoder, gaussian_state, splat_frame, render_gaussian_video


class WanGaussianTests(unittest.TestCase):
    def test_frames_are_orthonormal_and_scales_positive(self):
        torch.manual_seed(3)
        raw = torch.randn(2, 9, 4, 7)
        for field in (raw, torch.zeros_like(raw)):
            state = gaussian_state(field, 16, 28)
            axes = state['axes']
            error = axes.transpose(-1, -2) @ axes - torch.eye(2)
            self.assertLess(error.abs().max().item(), 1e-6)
            self.assertTrue((state['scale'] > 0).all())
            self.assertTrue((torch.linalg.det(axes) > .999).all())

    def test_rectangular_coverage_and_constant_colour(self):
        raw = torch.zeros(1, 9, 4, 7)
        image, coverage = splat_frame(raw, 16, 28)
        self.assertEqual(tuple(image.shape), (1, 3, 16, 28))
        self.assertTrue((coverage > 0).all())
        self.assertLess((image - .5).abs().max().item(), 1e-6)

    def test_orientation_changes_anisotropic_pixels(self):
        torch.manual_seed(11)
        raw = torch.randn(1, 9, 4, 7)
        raw[:, 2] = 3
        raw[:, 3] = -3
        raw[:, 4] = 1
        raw[:, 5] = 0
        rotated = raw.clone()
        rotated[:, 4] = 0
        rotated[:, 5] = 1
        a, _ = splat_frame(raw, 16, 28)
        b, _ = splat_frame(rotated, 16, 28)
        self.assertGreater((a-b).abs().mean().item(), .001)

    def test_decoder_only_generates_fields_and_backpropagates(self):
        torch.manual_seed(5)
        model = WanGaussianDecoder(width=8)
        latent = torch.randn(1, 16, 2, 2, 3, requires_grad=True)
        raw = model(latent)
        self.assertEqual(tuple(raw.shape), (1, 9, 5, 4, 6))
        video = render_gaussian_video(raw, 16, 24)
        self.assertEqual(tuple(video.shape), (1, 5, 3, 16, 24))
        video.square().mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
