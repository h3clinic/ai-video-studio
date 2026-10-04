import math
import unittest
import torch
from real_video.detail_lift import lift
from real_video.gaussian3d import project


class DetailLiftTests(unittest.TestCase):
    def setUp(self):
        self.u = torch.tensor([[30.5, 50.5], [170.5, 110.5]], dtype=torch.float64)
        self.s = torch.tensor([[[2., .3], [.3, 3.]], [[4., -.2], [-.2, 1.]]], dtype=torch.float64)
        self.d = torch.tensor([2., 5.], dtype=torch.float64)
        self.f = 120/math.tan(math.radians(42)/2)

    def test_projection_identity_and_positive_volume(self):
        xyz, cov = lift(self.u, self.s, self.d, self.f, (208, 120))
        mean, screen, z, _, _ = project(xyz, cov, self.u.new_zeros(3), self.u.new_tensor([0, 0, -1]), 240, 416)
        torch.testing.assert_close(mean, self.u)
        torch.testing.assert_close(screen, self.s, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(z, self.d)
        self.assertTrue((torch.linalg.eigvalsh(cov) > 0).all())

    def test_depth_ambiguity(self):
        a, _ = lift(self.u, self.s, self.d, self.f, (208, 120))
        b, _ = lift(self.u, self.s, self.d*2, self.f, (208, 120))
        torch.testing.assert_close(a*2, b)

    def test_reject_invalid(self):
        for d in [self.d*0, self.d*float('nan')]:
            with self.assertRaises(ValueError): lift(self.u, self.s, d, self.f, (208, 120))
        with self.assertRaises(ValueError): lift(self.u, self.s*.01, self.d, self.f, (208, 120))

    def test_gradients(self):
        d = self.d.clone().requires_grad_()
        xyz, cov = lift(self.u, self.s, d, self.f, (208, 120))
        (xyz.square().sum()+cov.square().sum()).backward()
        self.assertTrue(torch.isfinite(d.grad).all())

    def test_roundoff_and_real_asymmetry(self):
        cov = self.s.float()
        cov[0, 0, 1] += 1e-7
        lift(self.u.float(), cov, self.d.float(), self.f, (208, 120))
        cov[0, 0, 1] += .1
        with self.assertRaises(ValueError): lift(self.u.float(), cov, self.d.float(), self.f, (208, 120))

    def test_tangent_lift_preserves_projection(self):
        normals = self.u.new_tensor([[.2, .1, 1.], [-.3, .2, 1.]])
        xyz, cov = lift(self.u, self.s, self.d, self.f, (208, 120), normals=normals)
        mean, screen, _, _, _ = project(xyz, cov, self.u.new_zeros(3), self.u.new_tensor([0, 0, -1]), 240, 416)
        torch.testing.assert_close(mean, self.u)
        torch.testing.assert_close(screen, self.s, atol=1e-7, rtol=1e-7)


if __name__ == '__main__': unittest.main()
