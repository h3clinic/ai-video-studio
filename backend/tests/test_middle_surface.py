import unittest
import numpy as np
import torch
from real_video.middle_surface import middle_field,replacement_weight,distance_confidence,support_weights

class MiddleTests(unittest.TestCase):
    def test_confidence_bounded_and_monotone(self):
        d=np.linspace(0,.04,100);c=distance_confidence(d)
        self.assertEqual(c[0],1);self.assertEqual(c[-1],0)
        self.assertTrue((np.diff(c)<=0).all())
    def test_unsupported_patch_does_not_erase_original(self):
        old=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
        new=torch.tensor([[0.,0.,0.],[4.,0.,0.]])
        a,b=support_weights(old,new,torch.ones(2),torch.ones(2))
        torch.testing.assert_close(a,torch.tensor([1.,0.]))
        torch.testing.assert_close(b,torch.tensor([1.,0.]))
    def test_empty_patch_preserves_original(self):
        a,b=support_weights(torch.zeros(2,3),torch.empty(0,3),torch.ones(2),torch.empty(0))
        self.assertEqual(a.sum(),0);self.assertEqual(len(b),0)
    def test_only_two_boundaries_and_linear_interior(self):
        a=np.zeros((16,16),bool);a[4:12,4:12]=1;b=np.roll(a,2,axis=0)
        f=middle_field(a,b,9,.01)
        np.testing.assert_allclose(f[:,:,4],(f[:,:,0]+f[:,:,-1])/2,atol=1e-7)
        self.assertTrue(np.array_equal(f[:,:,0]<0,a));self.assertTrue(np.array_equal(f[:,:,-1]<0,b))
    def test_weight_zero_outside_positive_inside(self):
        p=torch.tensor([[2.,2.,0.],[2.,2.,1.],[0.,2.,0.]])
        w=replacement_weight(p,1,1,0,.5,.1)
        torch.testing.assert_close(w,torch.tensor([1.,0.,0.]))
    def test_empty_boundary_rejected(self):
        with self.assertRaises(ValueError):middle_field(np.zeros((4,4),bool),np.zeros((4,4),bool),5,.1)
