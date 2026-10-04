import unittest
import torch
from real_video.edit_gaussian_memory import affine, controls, splat_layer, transform


class GaussianEditTests(unittest.TestCase):
    def points(self):
        return torch.tensor([[.4,.5,-3.,-3.5,1.,0.,.8,.1,-.2,1.],
                             [.6,.6,-3.2,-3.,0.,1.,.4,.3,-.1,1.]])

    def test_identity_and_owned_source(self):
        points=self.points(); saved=points.clone(); pivot=torch.tensor([.5,.5])
        result=transform(points,pivot,controls(0,1))
        self.assertTrue(torch.allclose(result,points))
        result.add_(1)
        self.assertTrue(torch.equal(points,saved))

    def test_rotations_scales_and_mirrors_preserve_covariance(self):
        points=self.points(); pivot=torch.tensor([.5,.5])
        def covariance(p):
            u=p[:,4:6]; v=torch.stack((-u[:,1],u[:,0]),-1)
            rotation=torch.stack((u,v),-1)
            return rotation@torch.diag_embed((2*p[:,2:4]).exp())@rotation.transpose(-1,-2)
        for case in range(6):
            control=controls(case,1); matrix,_=affine(control,points)
            result=transform(points,pivot,control)
            expected=matrix@covariance(points)@matrix.T
            self.assertTrue(torch.allclose(covariance(result),expected,atol=1e-7))
            self.assertTrue(torch.allclose(result[:,4:6].norm(dim=-1),torch.ones(2)))

    def test_mirror_twice_recovers_source(self):
        points=self.points(); pivot=torch.tensor([.5,.5]); control=controls(3,1)
        result=transform(transform(points,pivot,control),pivot,control)
        self.assertTrue(torch.allclose(result,points,atol=1e-6))

    def test_grey_changes_colour_only(self):
        points=self.points(); result=transform(points,torch.tensor([.5,.5]),controls(4,1))
        self.assertTrue(torch.equal(result[:,:6],points[:,:6]))
        self.assertTrue(torch.allclose(result[:,6],result[:,7]))
        self.assertTrue(torch.allclose(result[:,7],result[:,8]))

    def test_layer_has_finite_bounded_alpha_and_offscreen_vanishes(self):
        points=self.points(); rgb,alpha=splat_layer(points,32,48,radius=5)
        self.assertTrue(torch.isfinite(rgb).all())
        self.assertTrue(((alpha>=0)&(alpha<=1)).all())
        points[:,:2]+=100
        rgb,alpha=splat_layer(points,32,48,radius=5)
        self.assertEqual(alpha.max().item(),0)
        self.assertEqual(rgb.max().item(),0)


if __name__=='__main__': unittest.main()
