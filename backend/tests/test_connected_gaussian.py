import unittest
import numpy as np
import torch
from real_video.connected_gaussian import build_surface,ConnectedSurface,deform_surface


class ConnectedGaussianTests(unittest.TestCase):
    def setUp(self):
        self.p=torch.zeros(25,10); self.p[:,:2]=torch.tensor([[x+.2,y+.3] for y in range(5) for x in range(5)])
        self.p[:,2:4]=-2; self.p[:,4]=1; self.p[:,9]=1
        self.asset=build_surface(self.p,1.)

    def test_identity(self):
        q=deform_surface(self.asset,self.asset['rest'])
        torch.testing.assert_close(q[:,:2],self.p[:,:2]); torch.testing.assert_close(q[:,6:],self.p[:,6:])

    def test_rigid_transform(self):
        r=torch.tensor([[0.,-1.],[1.,0.]])
        v=self.asset['rest']@r.T+torch.tensor([2.,3.])
        q=deform_surface(self.asset,v)
        torch.testing.assert_close(q[:,:2],self.p[:,:2]@r.T+torch.tensor([2.,3.]))
        torch.testing.assert_close(q[:,2:4],self.p[:,2:4])

    def test_rigid_arap(self):
        s=ConnectedSurface(self.asset); r=np.array([[0.,-1.],[1.,0.]])
        target=s.rest@r.T+[2.,3.]; v,stats=s.step(target)
        np.testing.assert_allclose(v,target,atol=1e-8); self.assertEqual(stats['accepted_fraction'],1.)

    def test_reject_collapse(self):
        with self.assertRaises(ValueError): deform_surface(self.asset,torch.zeros_like(self.asset['rest']))

    def test_stretch_guard(self):
        s=ConnectedSurface(self.asset); _,stats=s.step(s.rest*20)
        self.assertGreaterEqual(stats['min_determinant'],.2); self.assertLessEqual(stats['max_stretch'],2.2)

    def test_fixed_attachment_and_ownership(self):
        saved=self.asset['points'].clone(); self.p.zero_()
        self.assertTrue(torch.equal(self.asset['points'],saved))
        self.assertTrue((self.asset['bary']>=0).all()); self.assertTrue((self.asset['bary'].sum(1)<=1+1e-6).all())

    def test_resume(self):
        a=ConnectedSurface(self.asset); a.step(a.rest+[.1,.3]); b=ConnectedSurface(self.asset)
        b.previous=a.previous.copy()
        va,_=a.step(a.rest+[.2,.4]); vb,_=b.step(b.rest+[.2,.4]); np.testing.assert_array_equal(va,vb)

    def test_nonfinite_rejected(self):
        s=ConnectedSurface(self.asset)
        with self.assertRaises(ValueError): s.step(s.rest*np.nan)

    def test_covariance_transport(self):
        a=self.asset; a['points'][:,2:4]=torch.tensor([-2.,-3.])
        transform=torch.tensor([[1.2,.3],[.1,.8]])
        q=deform_surface(a,a['rest']@transform.T)
        u=q[:,4:6]; v=torch.stack((-u[:,1],u[:,0]),1); axes=torch.stack((u,v),-1)
        actual=(axes*q[:,2:4].exp().square()[:,None])@axes.transpose(-1,-2)
        expected=transform@torch.diag(torch.exp(torch.tensor([-4.,-6.])))@transform.T
        torch.testing.assert_close(actual,expected.expand_as(actual),atol=1e-7,rtol=1e-5)

    def test_local_guard_preserves_root_motion(self):
        s=ConnectedSurface(self.asset); v,_=s.step(s.rest*10+np.array([30.,10.]))
        self.assertGreater(np.linalg.norm(v.mean(0)-s.rest.mean(0)),20)
        self.assertGreaterEqual(s.quality(v)['min_stretch'],.45)

    def test_guarded_surface_is_shared(self):
        s=ConnectedSurface(self.asset); target=s.rest.copy(); target[0]+=[4,-4]
        v,_=s.step(target); q=s.quality(v)
        self.assertGreaterEqual(q['min_determinant'],.2); self.assertLessEqual(q['max_stretch'],2.2)
        # Vertex array, not independent per-triangle copies, defines all faces.
        self.assertEqual(len(v),len(self.asset['rest']))
