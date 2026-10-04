import unittest
import numpy as np
import torch
from real_video.learned_motion import LearnedGaussianMotion
from real_video.dense_gaussian_seed import FixedGeometrySplat
from real_video.demo_learned_motion import transfer_displacement,deform_covariance,sampled_jacobian
from real_video.gaussian_motion_memory import sample
from real_video.edit_gaussian_memory import splat_layer


class LearnedMotionTests(unittest.TestCase):
    def fields(self):
        f=torch.zeros(1,9,8,8); f[:,4]=1; f[:,2:4]=-4
        b=f.clone(); b[:,0]+=.005; c=b.clone(); c[:,0]+=.005
        return f,b,c

    def test_learned_rollout_is_differentiable(self):
        model=LearnedGaussianMotion(); s=model.initialize(*self.fields())
        for _ in range(3): s=model.step(s)
        s['fields'][:,:2].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_geometry_only_rollout_finite_fixed_shape(self):
        model=LearnedGaussianMotion(); source=self.fields(); s=model.initialize(*source)
        counts=[]
        with torch.no_grad():
            for _ in range(64):
                s=model.step(s,dt=.5)
                counts.append(sum(x.numel() for x in s.values()))
        self.assertEqual(len(set(counts)),1)
        self.assertTrue(torch.equal(s['fields'][:,2:],source[2][:,2:]))
        self.assertTrue(all(torch.isfinite(v).all() for v in s.values()))
        with self.assertRaises(ValueError): model.step(s,dt=0)

    def test_dense_motion_sampling(self):
        delta=torch.ones(1,2,8,8)*.1; uv=torch.tensor([[-1.,-1.],[0.,0.],[1.,1.]])
        result=transfer_displacement(delta,uv)
        self.assertTrue(torch.allclose(result,torch.ones(3,2)*.1))

    def test_fixed_operator_matches_existing_splat(self):
        p=torch.tensor([[.4,.5,-3.,-3.5,1.,0.,.7,.1,-.4,1.],[.6,.5,-3.,-3.5,0.,1.,.4,.6,-.4,1.]])
        rgb,alpha=splat_layer(p,32,48,radius=4)
        op=FixedGeometrySplat(p,32,48,radius=4)
        image,coverage=op.render((p[:,6:9]+1)/2)
        self.assertTrue(torch.allclose(rgb,image,atol=1e-6))
        self.assertTrue(torch.equal(alpha,coverage))

    def test_sampling_above_opencv_destination_limit(self):
        image=np.arange(64,dtype=np.float32).reshape(8,8)
        points=np.tile(np.array([[2.,3.]],np.float32),(40000,1))
        values=sample(image,points)
        self.assertEqual(values.shape,(40000,))
        self.assertTrue(np.all(values==26))

    def test_covariance_transport_and_sampled_identity(self):
        p=torch.tensor([[.4,.5,-3.,-3.5,1.,0.,.7,.1,-.4,1.]])
        rotation=torch.tensor([[[0.,-1.],[1.,0.]]])
        result=deform_covariance(p,rotation)
        self.assertTrue(torch.allclose(result[:,2:4],p[:,2:4],atol=1e-6))
        self.assertTrue(torch.allclose(result[:,4:6].abs(),torch.tensor([[0.,1.]]),atol=1e-6))
        yy,xx=torch.meshgrid((torch.arange(8)+.5)/8,(torch.arange(8)+.5)/8,indexing='ij')
        pos=torch.stack((xx,yy))[None]
        jac=sampled_jacobian(pos,torch.tensor([[0.,0.]]),aspect=1.)
        self.assertTrue(torch.allclose(jac,torch.eye(2)[None],atol=1e-6))


if __name__=='__main__': unittest.main()
