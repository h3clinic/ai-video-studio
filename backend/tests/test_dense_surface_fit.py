import unittest
import numpy as np
import torch
from real_video.dense_surface_fit import ARAPMesh, lift_pixels, source_pixels, geometry_metrics, sample


class DenseSurfaceFitTests(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,.8]],np.float32)
        self.f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,3,2],[0,2,1]])
        self.s=ARAPMesh(self.v,self.f)
    def test_rest_identity(self):
        result=self.s.solve(self.v,np.ones(5),iterations=3)
        np.testing.assert_allclose(result,self.v,atol=1e-6)
    def test_shared_translation(self):
        target=self.v+[.2,-.1,.3]
        result=self.s.solve(target,np.ones(5)*100,prior=1e-8,iterations=10)
        np.testing.assert_allclose(result,target,atol=1e-5)
    def test_no_data_is_rest(self):
        result=self.s.solve(self.v+2,np.zeros(5),iterations=3)
        np.testing.assert_allclose(result,self.v,atol=1e-6)
    def test_relative_weight_rest_identity(self):
        s=ARAPMesh(self.v,self.f,'relative')
        np.testing.assert_allclose(s.solve(self.v,np.ones(5),iterations=3),self.v,atol=1e-6)
    def test_translated_prior(self):
        target=self.v+[.2,-.1,.3]
        result=self.s.solve(target,np.zeros(5),prior=1.,iterations=10,prior_target=target)
        np.testing.assert_allclose(result,target,atol=1e-5)
    def test_invalid_confidence(self):
        with self.assertRaises(ValueError): self.s.solve(self.v,np.ones(5)*-1)
    def test_camera_lift_round_trip(self):
        c=dict(eye=torch.tensor([0.,0.,3.]),target=torch.zeros(3),fov=40.,width=512,height=512,size=607,pad_x=45,pad_y=141,x0=0,y0=106)
        p=source_pixels(self.v,c); back=lift_pixels(p,self.v,c)
        np.testing.assert_allclose(back,self.v,atol=1e-6)
    def test_strain_detects_stretch(self):
        moved=self.v.copy(); moved[:,0]*=10
        g=geometry_metrics(self.v,moved,self.f,self.s.edges)
        self.assertGreater(g['edge_ratio_p99'],5.)
    def test_sample_vector_field(self):
        field=np.ones((10,10,2),np.float32)*[2,3]
        np.testing.assert_allclose(sample(field,np.array([[2,4],[6,7]])),[[2,3],[2,3]])


if __name__=='__main__': unittest.main()
