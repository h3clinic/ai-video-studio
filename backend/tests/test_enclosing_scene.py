import unittest
import numpy as np
import torch
from real_video.enclosing_scene import rays,pano_directions,sample_pano,integrate,VIEWS
from real_video.gaussian3d import look_at

class EnclosingSceneTests(unittest.TestCase):
    def test_six_centers_and_unit_rays(self):
        for forward,up in VIEWS.values():
            rr=rays(33,90,forward,up)
            np.testing.assert_allclose(rr[16,16],forward,atol=1e-7)
            np.testing.assert_allclose(np.linalg.norm(rr,axis=-1),1,atol=1e-7)

    def test_six_overlapping_views_cover_sphere(self):
        p=np.zeros((64,128,3),np.uint8);known=np.zeros((64,128),bool);d=pano_directions(64)
        for f,u in VIEWS.values(): integrate(p,known,np.full((32,32,3),120,np.uint8),d,110,f,u)
        self.assertTrue(known.all());self.assertEqual(int(p.min()),120)

    def test_existing_evidence_never_overwritten(self):
        p=np.full((32,64,3),43,np.uint8);known=np.ones((32,64),bool)
        integrate(p,known,np.full((32,32,3),200,np.uint8),pano_directions(32),110,[0,0,-1],[0,1,0])
        self.assertEqual(int(p.max()),43)

    def test_pole_camera_orthonormal(self):
        for y in [-1.,1.]:
            m=look_at(torch.tensor([0.,y,0.]),torch.zeros(3))
            torch.testing.assert_close(m@m.T,torch.eye(3))
        with self.assertRaises(ValueError):look_at(torch.zeros(3),torch.zeros(3))

    def test_constant_panorama_sampling(self):
        p=np.full((32,64,3),.37,np.float32)
        for f,u in VIEWS.values():np.testing.assert_allclose(sample_pano(p,rays(32,110,f,u)),.37,atol=1e-6)
