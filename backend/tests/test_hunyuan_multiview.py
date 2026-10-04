import unittest
import torch
from real_video.hunyuan_multiview import camera,canonical,rasterize,screen_position,VIEWS


class MultiviewControlsTests(unittest.TestCase):
    def test_all_camera_frames_including_poles(self):
        for e,a in VIEWS:
            eye,r=camera(e,a,'cpu')
            torch.testing.assert_close(r@r.T,torch.eye(3),atol=1e-6,rtol=1e-6)
            self.assertAlmostEqual(float(torch.det(r)),1.,places=5)
            self.assertTrue(torch.isfinite(screen_position(torch.zeros(1,3),e,a,32)).all())

    def test_nearest_triangle_and_barycentric(self):
        near=torch.tensor([[1.,1.,1.],[6.,1.,1.],[1.,6.,1.]])
        far=near.clone();far[:,2]=2
        face,bary,depth=rasterize(torch.cat((far,near)),torch.tensor([[0,1,2],[3,4,5]]),8)
        self.assertEqual(int(face[2,2]),1)
        self.assertAlmostEqual(float(depth[2,2]),1.)
        torch.testing.assert_close(bary[2,2].sum(),torch.tensor(1.))
        point=(near*bary[2,2,:,None]).sum(0)
        torch.testing.assert_close(point,torch.tensor([2.5,2.5,1.]))
        self.assertEqual(int(face[7,7]),-1)

    def test_canonical_scale(self):
        v=torch.tensor([[-1.,-2.,-1.],[1.,2.,1.]])
        p,_,_=canonical(v)
        self.assertAlmostEqual(float(p.norm(dim=1).max()*2),1.15,places=5)
