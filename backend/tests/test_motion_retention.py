import unittest
import torch
from real_video.retain_coherent_motion import blend


class MotionRetentionTests(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        a={'x':torch.tensor([1.,2.])}; b={'x':torch.tensor([3.,4.])}
        self.assertTrue(torch.equal(blend(a,b,0)['x'],a['x']))
        self.assertTrue(torch.equal(blend(a,b,1)['x'],b['x']))
        self.assertTrue(torch.equal(blend(a,b,.5)['x'],torch.tensor([2.,3.])))

    def test_does_not_mutate_sources(self):
        a={'x':torch.tensor([1.,2.])}; b={'x':torch.tensor([3.,4.])}
        result=blend(a,b,.5); result['x'].fill_(0)
        self.assertTrue(torch.equal(a['x'],torch.tensor([1.,2.])))
        self.assertTrue(torch.equal(b['x'],torch.tensor([3.,4.])))

    def test_rejects_incompatible_checkpoints(self):
        a={'x':torch.zeros(2)}
        for alpha in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError): blend(a,a,alpha)
        with self.assertRaises(ValueError): blend(a,{'y':torch.zeros(2)},.5)
        with self.assertRaises(ValueError): blend(a,{'x':torch.zeros(3)},.5)


if __name__=='__main__': unittest.main()
